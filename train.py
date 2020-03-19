
#coding=utf-8

import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from naoqi import ALProxy

import matplotlib.pyplot as plt
import numpy as np
import time
import math

#buffer 格式
Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'next_state', 'reward'))
EPS = 1e-10

#action 19个关节角度改变值
names = [
        "HeadPitch", \
    #"HeadYaw",  \
    "RShoulderRoll", "RShoulderPitch", "RElbowYaw", "RElbowRoll", \
    #    "RWristYaw",  "RHand", \
    "LShoulderRoll", "LShoulderPitch", "LElbowYaw", "LElbowRoll", \
    #    "LWristYaw", "LHand", \
    "RHipPitch", "RHipRoll", "RKneePitch", "RAnklePitch", "RAnkleRoll", \
    "LHipYawPitch", "LHipPitch", "LHipRoll", "LKneePitch", "LAnklePitch", "LAnkleRoll"]

#存放参数类
class args(object):
    env_name = 'Nao'
    seed = 1234
    num_episode = 1000 # 总迭代次数
    batch_size = 6144 # 每个episode 的step数
    max_step_per_round = 512 # 每个 trajectory 的step数
    gamma = 0.9 # 衰减项
    lamda = 0.97
    log_num_episode = 1 # 每隔10次保存一次模型并输出一些信息
    num_epoch = 10 # 每个episodeb遍历所有trajecory的次数
    minibatch_size = 512 #一次更新的step数
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.00
    lr = 3e-4
    num_parallel_run = 1 #

    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = True
    advantage_norm = True
    lossvalue_norm = True


    # nao relative information
    state_num = 35
    action_num = 20
    r_range = 60 # reward 范围大小
    a_range = 0.3 # action 范围大小
    a_clip = 0.3
    ip = "127.0.0.1"
    port = 9559

    experts =False


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

# state 归一化
class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_norm=True):
        super(ActorCritic, self).__init__()

        self.actor_fc1 = nn.Linear(num_inputs, 256)
        self.actor_fc2 = nn.Linear(256, 128)
        self.actor_fc3 = nn.Linear(128, num_outputs)
        self.actor_logstd = nn.Parameter(torch.full((1, num_outputs) , -0.6))#标准差设为固定的值

        self.critic_fc1 = nn.Linear(num_inputs, 256)
        self.critic_fc2 = nn.Linear(256, 128)
        self.critic_fc3 = nn.Linear(128, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = torch.tanh(self.actor_fc3(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


def ppo(args):
    env = nao_env()
    num_inputs = args.state_num
    num_actions = args.action_num

    torch.manual_seed(args.seed)

    network = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm)

    # 读入上次训练好的参数

    try:
        network.load_state_dict(torch.load('params.pkl'))
    except Exception:
        print(' model load wrong!')
    else:
        print("model load sucessfully!")

    optimizer = opt.Adam(network.parameters(), lr=args.lr)

    running_state = ZFilter((num_inputs,), clip=10.0)


    # record actorLoss and criticLoss for plot

    actorLoss_record = []
    criticLoss_record = []

    # record average 1-round cumulative reward in every episode
    reward_record = []
    global_steps = 0

    lr_now = args.lr
    clip_now = args.clip

    for i_episode in range(args.num_episode):
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        memory = Memory()
        num_steps = 0
        reward_list = []
        len_list = []
        while num_steps < args.batch_size:
            if args.experts:
                state,joints = env.getStates()
                if args.state_norm:
                    state = running_state(state)
                reward_sum = 0
                char = ' '
                if char == 'z':
                    args.experts = False
                for t in range(args.max_step_per_round):
                    if char != 'q':
                        action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0))
                        char = raw_input("input:")
                        next_state,newjoints = env.getStates()

                        action = newjoints - joints
                        action1, logproba = network.select_action(Tensor(action), action_logstd)
                        action = np.clip(action, -args.a_clip, args.a_clip)
                        action =action.tolist()
                        logproba = logproba.data.numpy()[0]
                        next_state, reward, done = env.step(action)

                        if args.state_norm:
                            next_state = running_state(next_state)

                        memory.push(state, value, action, logproba, next_state, reward)
                    else:
                        action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0))
                        action, logproba = network.select_action(action_mean, action_logstd)
                        action = action.data.numpy()[0]  # tensor -> numpy
                        action = action * args.a_range

                        action = np.clip(action, -args.a_clip, args.a_clip)  # 限制动作大小
                        '''
                        random =  np.random.normal(0, 0.1*args.a_range, args.action_num)
                        action = action + random
                        '''
                        action = action.tolist()  # numpy -> list
                        # print("action",action)
                        logproba = logproba.data.numpy()[0]
                        next_state, reward, done = env.step(action)
                        # print("step" , t , " reward",reward)
                        reward_sum += reward
                        if args.state_norm:
                            next_state = running_state(next_state)
                        memory.push(state, value, action, logproba, next_state, reward)
                    if done:
                        break

                    state = next_state
                    joints = newjoints
            else :
                state = env.reset()
                if args.state_norm:
                    state = running_state(state)
                reward_sum = 0
                for t in range(args.max_step_per_round):
                    action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0))
                    action, logproba = network.select_action(action_mean, action_logstd)
                    action =  action.data.numpy()[0] # tensor -> numpy
                    action = action * args.a_range

                    action = np.clip(action, -args.a_clip, args.a_clip) # 限制动作大小
                    '''
                    random =  np.random.normal(0, 0.1*args.a_range, args.action_num)
                    action = action + random
                    '''
                    action = action.tolist() # numpy -> list
                   # print("action",action)
                    logproba = logproba.data.numpy()[0]
                    next_state, reward, done= env.step(action)
                   # print("step" , t , " reward",reward)
                    reward_sum += reward
                    if args.state_norm:
                        next_state = running_state(next_state)

                    memory.push(state, value, action, logproba, next_state, reward)

                    if done:
                        break

                    state = next_state

            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_list.append(reward_sum)
            len_list.append(t + 1)
            print("sampling now:",num_steps," mean trajectory reward:",np.mean(reward_list))

        reward_record.append({
            'episode': i_episode,
            'steps': global_steps,
            'meanepreward': np.mean(reward_list),
            'meaneplen': np.mean(len_list)})

        batch = memory.sample()
        batch_size = len(memory)
        num_steps = 0

        # step2: extract variables from trajectories
        rewards = Tensor(batch.reward)
        values = Tensor(batch.value)
        actions = Tensor(batch.action)
        states = Tensor(batch.state)
        oldlogproba = Tensor(batch.logproba)

        returns = Tensor(batch_size)
        deltas = Tensor(batch_size)
        advantages = Tensor(batch_size)

       # prev_return = 0
        prev_value = 0
        prev_advantage = 0
        while num_steps < batch_size :
            prev_value = values[num_steps + args.max_step_per_round - 1]
            prev_return = prev_value
            for i in reversed(range(num_steps,num_steps+args.max_step_per_round)):
                returns[i] = rewards[i] + args.gamma * prev_return
                deltas[i] = rewards[i] + args.gamma * prev_value  - values[i]
                # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
                advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage

                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            num_steps += args.max_step_per_round
        if args.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
            print("optimize now, round ",i_epoch)
            # sample from current batch
            minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
            minibatch_states = states[minibatch_ind]
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_newlogproba = network.get_logproba(minibatch_states, minibatch_actions)
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = network._forward_critic(minibatch_states).flatten()

            ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
            loss_surr = - torch.mean(torch.min(surr1, surr2))

            # not sure the value loss should be clipped as well
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
            # moreover, original paper does not mention clipped value
            if args.lossvalue_norm:
                minibatch_return_6std = 6 * minibatch_returns.std()
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
            else:
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

            loss_entropy=0
           # loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

            total_loss = loss_surr + args.loss_coeff_value * loss_value  # + args.loss_coeff_entropy * loss_entropy
            print("episode:",i_episode," i_epoch:",i_epoch,\
                  " aloss:",loss_surr.data.numpy().tolist()," closs:",loss_value.data.numpy().tolist(),\
                  " total_loss:",total_loss.data.numpy().tolist())
            actorLoss_record.append(loss_surr.data.numpy().tolist())
            criticLoss_record.append(loss_value.data.numpy().tolist())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if args.schedule_clip == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            clip_now = args.clip * ep_ratio

        if args.schedule_adam == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            lr_now = args.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in optimizer.param_groups:
                g['lr'] = lr_now

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} Reward: {:.4f} total_loss = {:.4f} = {:.4f} + {} * {:.4f} ' \
                  .format(i_episode, reward_record[-1]['meanepreward'], total_loss.data, loss_surr.data,
                          args.loss_coeff_value,
                          loss_value.data))
            print('-----------------')

            # 保存模型
            try:
                torch.save(network.state_dict(), 'params.pkl')
            except Exception:
                print(' model saved wrong!')
            else:
                print("model saved sucessfully!")


    return reward_record,actorLoss_record,criticLoss_record



class nao_env(object):
    def __init__(self):
        self.motionProxy = ALProxy("ALMotion", args.ip, args.port)
        self.postureProxy = ALProxy("ALRobotPosture", args.ip, args.port)
        self.memoryProxy = ALProxy ("ALMemory", args.ip, args.port)
        self.sonarProxy = ALProxy("ALSonar", args.ip, args.port)
        self.sonarProxy.subscribe("myApplication")

        self.motionProxy.setFallManagerEnabled(False)
        self.motionProxy.setStiffnesses(names, 1.0)

        self.fractionMaxSpeed = 1.0
        # self.postureProxy.goToPosture('StandInit', 0.5)
        #self.motionProxy.setAngles(names, self.dest_state, 0.05)
        #time.sleep(1)
    def reset(self):

        # 初始化 平躺

        print "Resetting"
        flag = self.postureProxy.goToPosture('Stand', 1)
        print "Reset Success: ", flag
        '''
        action = np.array([0.0]*args.action_num)
        random =  np.random.normal(0, args.a_range, args.action_num)
        action = action + random
        action = action.tolist()
        self.motionProxy.changeAngles(names, action, self.fractionMaxSpeed)
        '''

        useSensors = True
        states=[]

        # state 包括 19个关节角度 8个足部压力传感器 2个身体角度
        JointAngles = self.motionProxy.getAngles(names, useSensors)
        LFrontL = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/FrontLeft/Sensor/Value")
        LFrontR = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/FrontRight/Sensor/Value")
        LRearL = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value")
        LRearR = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value")
        RFrontL = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/FrontLeft/Sensor/Value")
        RFrontR = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/FrontRight/Sensor/Value")
        RRearL = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value")
        RRearR = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/RearRight/Sensor/Value")
        FootSensor = [LFrontL, LFrontR, LRearL, LRearR, RFrontL, RFrontR, RRearL, RRearR]
        angleX = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
        angleY = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
        sonarLeft = self.memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value")
        sonarRigt = self.memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value")
        aX = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value")
        aY = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value")
        aZ = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value")
        states += JointAngles
        states += FootSensor
        states += [angleX, angleY]
        states += [sonarLeft, sonarRigt]
        states += [aX, aY, aZ]
        return np.array(states)

    def getStates(self):

        '''
        print "waiting~"
        time.sleep(30)
        print "waitiny done!"
        '''
        useSensors = True
        states=[]

        # state 包括 19个关节角度 8个足部压力传感器 2个身体角度
        JointAngles = self.motionProxy.getAngles(names, useSensors)
        LFrontL = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/FrontLeft/Sensor/Value")
        LFrontR = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/FrontRight/Sensor/Value")
        LRearL = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value")
        LRearR = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value")
        RFrontL = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/FrontLeft/Sensor/Value")
        RFrontR = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/FrontRight/Sensor/Value")
        RRearL = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value")
        RRearR = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/RearRight/Sensor/Value")
        FootSensor = [LFrontL, LFrontR, LRearL, LRearR, RFrontL, RFrontR, RRearL, RRearR]
        angleX = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
        angleY = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
        sonarLeft = self.memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value")
        sonarRigt = self.memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value")
        aX = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value")
        aY = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value")
        aZ = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value")
        states += JointAngles
        states += FootSensor
        states += [angleX, angleY]
        states += [sonarLeft, sonarRigt]
        states += [aX, aY, aZ]
        return np.array(states),np.array(JointAngles)

    def step(self, change):

        useSensors = True
        states = []
        reward = 0.0
        # done always be false
        done = False
        #self.motionProxy.setStiffnesses(names, 1.0)
        self.motionProxy.changeAngles(names, change, self.fractionMaxSpeed)
        # state 包括 19个关节角度 8个足部压力传感器 2个身体角度
        time.sleep(0.05)
        JointAngles = self.motionProxy.getAngles(names, useSensors)
        LFrontL = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/FrontLeft/Sensor/Value")
        LFrontR = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/FrontRight/Sensor/Value")
        LRearL = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/RearLeft/Sensor/Value")
        LRearR = self.memoryProxy.getData("Device/SubDeviceList/LFoot/FSR/RearRight/Sensor/Value")
        RFrontL = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/FrontLeft/Sensor/Value")
        RFrontR = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/FrontRight/Sensor/Value")
        RRearL = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/RearLeft/Sensor/Value")
        RRearR = self.memoryProxy.getData("Device/SubDeviceList/RFoot/FSR/RearRight/Sensor/Value")
        FootSensor = [LFrontL, LFrontR, LRearL, LRearR, RFrontL, RFrontR, RRearL, RRearR]
        angleX = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleX/Sensor/Value")
        angleY = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AngleY/Sensor/Value")
        sonarLeft = self.memoryProxy.getData("Device/SubDeviceList/US/Left/Sensor/Value")
        sonarRigt = self.memoryProxy.getData("Device/SubDeviceList/US/Right/Sensor/Value")
        aX = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerX/Sensor/Value")
        aY = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerY/Sensor/Value")
        aZ = self.memoryProxy.getData("Device/SubDeviceList/InertialSensor/AccelerometerZ/Sensor/Value")

        states += JointAngles
        states += FootSensor
        states += [angleX, angleY]
        states += [sonarLeft, sonarRigt]
        states += [aX, aY, aZ]
        #站立时足部8个压力传感器应该有较大的值
        reward1 = np.sum(FootSensor)

        #站立时足部8个压力传感器数值应该比较均匀
        reward2 = -np.var(FootSensor)

        #站立时身体在x和y方向的偏离度应该都为0
        # reward3 = -(angleX**2+angleY**2)

        #惩罚双脚悬空
        div1 = max(0.02,np.median([LFrontL, LFrontR, LRearL, LRearR]))
        div2 = max(0.02,np.median([RFrontL, RFrontR, RRearL, RRearR]))
        reward4 = -0.4/div1 -0.4/div2
        #直立时x/y/z加速度应为(0,0,-9.8)
        reward5 = -0.2*(aX**2+aY**2)-1.5*aZ
        reward = reward1 + reward2 + reward4 +reward5
        print (reward1, reward2, reward4, reward5, "aLL", reward)

        return np.array(states), reward, done

if __name__ == '__main__':
    args.env_name='Nao'
    reward_record,actorLoss_record,criticLoss_record = ppo(args)
    print(reward_record)
    plt.plot(np.array((range(len(actorLoss_record)))),actorLoss_record,color='r',label='actorLoss')
    plt.plot(np.array((range(len(criticLoss_record)))),criticLoss_record,color='b',label='criticLoss')
    plt.legend(['actorLoss','criticLoss'])
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.show()


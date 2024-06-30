import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import collections
import random


# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer():
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)
    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state):
        
        self.buffer.append((state, action, reward, next_state))
    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state = zip(*transitions)
        return state, action, reward, next_state
    # 目前队列长度
    def size(self):
        return len(self.buffer)

def init_weights(layer):
    # 如果为卷积层，使用正态分布初始化
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

class Actor(nn.Module):
    '''
    演员Actor网络
    '''
    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels=3,   # 指定入参数据的通道数。RGB图像是三个图层，即3通道
                out_channels=32,  # 指定该卷积层的输出通道数，即该层卷积核个数
                kernel_size=9,   # 指定卷积核尺寸，此时为5*5
                stride=1         # 指定卷积步长，即每做一次卷积后，卷积核平移一个像素
                )
        self.conv2 = nn.Conv2d(
                in_channels=32,   # 指定入参数据的通道数。RGB图像是三个图层，即3通道
                out_channels=64,  # 指定该卷积层的输出通道数，即该层卷积核个数
                kernel_size=5,   # 指定卷积核尺寸，此时为5*5
                stride=1         # 指定卷积步长，即每做一次卷积后，卷积核平移一个像素
                )
        self.conv3 = nn.Conv2d(
                in_channels=64,   # 指定入参数据的通道数。RGB图像是三个图层，即3通道
                out_channels=128,  # 指定该卷积层的输出通道数，即该层卷积核个数
                kernel_size=3,   # 指定卷积核尺寸，此时为5*5
                stride=1         # 指定卷积步长，即每做一次卷积后，卷积核平移一个像素
                )

        self.l1 = nn.Linear(128,64)
        self.l2 = nn.Linear(64,5)
        self.lstm = nn.LSTM(128, 128, 1, batch_first=True)

    def forward(self, x):
        o1 = self.conv1(x)
        f1 = F.relu(o1)
        o2 = self.conv2(f1)
        f2 = F.relu(o2)
        o3 = self.conv3(f2)
        f3 = F.relu(o3)
        f3 = torch.reshape(f3,(1,128))

        #r = self.lstm(f3)
        #全连接层
        o4 = self.l1(f3)
        f4 = F.relu(o4)
        o5 = self.l2(f4)
        out = F.softmax(o5,dim=1)
        return out


class Critic(nn.Module):
    '''
    评论家Critic网络
    '''
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels=3,   # 指定入参数据的通道数。RGB图像是三个图层，即3通道
                out_channels=32,  # 指定该卷积层的输出通道数，即该层卷积核个数
                kernel_size=9,   # 指定卷积核尺寸，此时为5*5
                stride=1         # 指定卷积步长，即每做一次卷积后，卷积核平移一个像素
                )
        self.conv2 = nn.Conv2d(
                in_channels=32,   # 指定入参数据的通道数。RGB图像是三个图层，即3通道
                out_channels=64,  # 指定该卷积层的输出通道数，即该层卷积核个数
                kernel_size=5,   # 指定卷积核尺寸，此时为5*5
                stride=1         # 指定卷积步长，即每做一次卷积后，卷积核平移一个像素
                )
        self.conv3 = nn.Conv2d(
                in_channels=64,   # 指定入参数据的通道数。RGB图像是三个图层，即3通道
                out_channels=128,  # 指定该卷积层的输出通道数，即该层卷积核个数
                kernel_size=3,   # 指定卷积核尺寸，此时为5*5
                stride=1         # 指定卷积步长，即每做一次卷积后，卷积核平移一个像素
                )

        self.l1 = nn.Linear(128,64)
        self.l2 = nn.Linear(64,1)
    def forward(self, x):
        o1 = self.conv1(x)
        f1 = F.relu(o1)
        o2 = self.conv2(f1)
        f2 = F.relu(o2)
        o3 = self.conv3(f2)
        f3 = F.relu(o3)
        f3 = torch.reshape(f3,(1,128))
        o4 = self.l1(f3)
        f4 = F.relu(o4)
        out = self.l2(f4)

        return out


class Actor_Critic:
    def __init__(self):
        self.gamma = 0.99
        self.lr_a = 1e-4
        self.lr_c = 1e-3


        self.action_dim = 5       #获取描述行动的数据维度
        #self.state_dim = self.env.observation_space.shape[0]  #获取描述环境的数据维度

        self.actor = Actor().cuda()   #创建演员网络
        self.critic = Critic().cuda()                  #创建评论家网络
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        self.loss = nn.MSELoss().cuda()
        # 深度集成参数
        self.ensemble_list = []
        self.optim_list = []
        self.critic_loss = []
        self.Ne = 5
        # for i in range(self.Ne):
        #     tmp_critic = Critic().cuda()
        #     tmp_critic.apply(init_weights)
        #     self.ensemble_list.append(tmp_critic)
        # for i in range(self.Ne):
        #     ensemble_optim = torch.optim.Adam(self.ensemble_list[i].parameters(), lr=self.lr_c)
        #     self.optim_list.append(ensemble_optim)


    def get_action(self, input):
        input = input.cuda()
        a = self.actor(input)

        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率
        action = action.cpu()
        return action.detach().numpy(), log_prob

    def learn(self, log,s,s_,rewards):#)
        #使用Critic网络估计状态值
        s = s.cuda()
        s_ = s_.cuda()
        v = self.critic(s)
        v_ = self.critic(s_)

        critic_loss = self.loss(self.gamma * v_ + rewards, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        td = self.gamma * v_ + rewards - v          #计算TD误差
        loss_actor = -log * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
    def deep_ensemble_learn(self,log,s,s_,rewards):
        v_total = 0
        v_total_ = 0
        critic_loss = 0
        s = s.cuda()
        s_ = s_.cuda()
        for model in self.ensemble_list:
            vi=model(s)
            vi_=model(s_)
            #self.v_1.append(vi_)
            # TD误差
            v_total = v_total + vi
            v_total_ = v_total_ + vi_
            self.critic_loss.append(self.loss(self.gamma*vi_+rewards,vi))
            critic_loss  = critic_loss + self.loss(self.gamma*vi_+rewards,vi)
        # for i in range(self.Ne):
        #     self.sigma = self.sigma + pow(self.v_1[i] - self.miu,2)/self.Ne

        
        for i in range(self.Ne):
            self.optim_list[i].zero_grad()

        critic_loss.backward()

        for i in range(self.Ne):
            self.optim_list[i].step()
        


        td = self.gamma * v_total_/self.Ne + rewards - v_total / self.Ne         #计算TD误差
        #td = critic_loss / self.Ne
        loss_actor = -log * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()


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
    def add(self, state, action, reward, next_state, ):
        
        self.buffer.append((state, action, reward, next_state))
    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)
    # 目前队列长度
    def size(self):
        return len(self.buffer)



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


    def forward(self, x):
        o1 = self.conv1(x)
        f1 = F.relu(o1)
        o2 = self.conv2(f1)
        f2 = F.relu(o2)
        o3 = self.conv3(f2)
        f3 = F.relu(o3)
        f3 = torch.reshape(f3,(1,128))


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
        # states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # log = torch.tensor(transition_dict['log']).view(-1,1)
        # rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1)
        # next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)

        # v = self.critic(states)
        # v_ = self.critic(next_states)


        critic_loss = self.loss(self.gamma * v_ + rewards, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        td = self.gamma * v_ + rewards - v          #计算TD误差
        loss_actor = -log * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
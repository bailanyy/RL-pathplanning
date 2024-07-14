import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import collections
import random
import math

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
        nn.init.normal_(layer.weight, mean=0, std=0.1)
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0, std=0.1)
        nn.init.normal_(layer.bias, mean=0, std=0.5)
        # nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        # nn.init.constant_(layer.bias, 0.1)

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


class Explorer(nn.Module):
    def __init__(self):
        super(Explorer, self).__init__()
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




class Q_Critic(nn.Module):
    '''
    评论家Critic网络
    '''
    def __init__(self):
        super(Q_Critic, self).__init__()
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
    
        self.crossEntropyLoss = nn.CrossEntropyLoss()


    def get_action(self, input):
        input = input.cuda()
        a = self.actor(input)

        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率
        action = action.cpu()
        return action.detach().numpy(), log_prob,a
    
    def get_next(self,input):
        self.all_next = input
    def get_crossentryloss(self,action_prob):
        x = self.get_posterior_distribution().cuda()
        action_prob = torch.Tensor(action_prob).cuda()
        loss = self.crossEntropyLoss(x, action_prob)
        return loss

    def get_posterior_distribution(self):
        v_value = np.zeros((1,5))
        
        for i in range(5):
            s = self.all_next[i].cuda()
            v = self.critic(s)
            v = v.cpu().detach().float()
            v_value[0][i] = v

        v_value = torch.Tensor(v_value)
        distibution = F.softmax(v_value)
        return distibution


        return distibution
    def learn(self, log,s,s_,rewards):#)
        #使用Critic网络估计状态值
        s = s.cuda()
        s_ = s_.cuda()
        v = self.critic(s)
        v_ = self.critic(s_)
        print(v)
        critic_loss = self.loss(self.gamma * v_ + rewards, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        td = self.gamma * v_ + rewards - v          #计算TD误差
        loss_actor = -log * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
 

# Q-base A2C with deep ensemble
# class A2C_QDE:
#     def __init__(self):
#         self.gamma = 0.99
#         self.lr_a = 1e-4
#         self.lr_c = 1e-3
#         self.Ne = 10

#         self.action_dim = 5       #获取描述行动的数据维度
        

#         self.actor = Actor().cuda()   #创建演员网络
#         self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)

#         self.ensemble_list = []
#         self.optim_list = []
#         self.critic_loss = []
#         for i in range(self.Ne):
#             tmp_critic = Q_Critic().cuda()
#             tmp_critic.apply(init_weights)
#             self.ensemble_list.append(tmp_critic)
#         for i in range(self.Ne):
#             ensemble_optim = torch.optim.Adam(self.ensemble_list[i].parameters(), lr=self.lr_c)
#             self.optim_list.append(ensemble_optim)
        
        
#         self.loss = nn.MSELoss().cuda()
#     def get_action(self, input):
#         input = input.cuda()
#         a = self.actor(input)

#         dist = Categorical(a)
#         action = dist.sample()             #可采取的action
#         log_prob = dist.log_prob(action)   #每种action的概率
#         action = action.cpu()
#         return action.detach().numpy(), log_prob,a

#     def learn(self, log,s,s_,rewards):#)
#         #使用Critic网络估计状态值
#         s = s.cuda()
#         s_ = s_.cuda()
#         # v = self.critic(s)
#         # v_ = self.critic(s_)
#         a,log,p = self.get_action(s)
#         a_,log_,p_ = self.get_action(s_)
#         #print(p_)
#         critic_loss = 0
#         for model in self.ensemble_list:
#             q=model(s)
#             q_=model(s_)
#             # q = q.cpu().detach().numpy()
#             # q_ = q_.cpu().detach().numpy()
#             q_t = q[0][a]
#             #print(q[0][a])
#             # TD误差
#             V_star = q_[0][0]*p_[0][0] + q_[0][1]*p_[0][1] + q_[0][2]*p_[0][2] + q_[0][3]*p_[0][3] + q_[0][4]*p_[0][4] 
#             #print(self.gamma*V_star+rewards,q[0][a])
#             self.critic_loss.append(self.loss(self.gamma*V_star+rewards,q[0][a]))
#             critic_loss  = critic_loss + self.loss(self.gamma*V_star+rewards,q[0][a])
        
#         for i in range(self.Ne):
#             self.optim_list[i].zero_grad()

#         critic_loss.backward()

#         for i in range(self.Ne):
#             self.optim_list[i].step()
        
#         td = critic_loss / self.Ne       #计算TD误差
#         loss_actor = -log * td.detach()
#         self.actor_optim.zero_grad()
#         loss_actor.backward()
#         self.actor_optim.step()







#   正常A2C
class A2C_DE:
    def __init__(self):
        self.gamma = 0.99
        self.lr_a = 1e-4
        self.lr_c = 1e-3
        self.lr_e = 1e-4
        self.Ne = 5

        self.action_dim = 5       #获取描述行动的数据维度
        

        self.actor = Actor().cuda()   #创建演员网络
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.explorer = Explorer().cuda()
        self.explorer_optim = torch.optim.Adam(self.explorer.parameters(), lr=self.lr_e)
        
        self.ensemble_list = []
        self.optim_list = []
        self.critic_loss = []
        self.all_next = []
        for i in range(self.Ne):
            tmp_critic = Critic().cuda()
            tmp_critic.apply(init_weights)
            self.ensemble_list.append(tmp_critic)
        # for i in range(self.Ne):
        #     ensemble_optim = torch.optim.Adam(self.ensemble_list[i].parameters(), lr=self.lr_c)
        #     self.optim_list.append(ensemble_optim)
        self.optimizer = torch.optim.Adam([{"params": model.parameters()} for model in self.ensemble_list], lr=self.lr_c)
        
        self.loss = nn.MSELoss().cuda()
        self.crossEntropyLoss = nn.CrossEntropyLoss()
    def get_next(self,input):
        self.all_next = input
    def get_crossentryloss(self,action_prob):
        x = self.get_posterior_distribution().cuda()
        action_prob = torch.Tensor(action_prob).cuda()
        loss = self.crossEntropyLoss(x, action_prob)
        return loss

    def get_posterior_distribution(self):
        mean = np.zeros((1,5))
        sigma = np.zeros((1,5))
        sigma_norm = np.zeros((1,5))
        posterior = np.zeros((1,5))
        v_value = np.zeros((5,self.Ne))
        for i in range(5):
            v_total = 0
            j=0
            for model in self.ensemble_list:
                s = self.all_next[i].cuda()
                v = model(s)
                v = v.cpu().detach().float()
                v_value[i][j] = v
                v_total += v
                j += 1
            v_mean = v_total / self.Ne
            #print(i,v_mean)
            mean[0][i] = v_mean
        #print(v_value)
       
        for i in range(5):
            v_sigma = 0
            sigma_total = 0
            for j in range(self.Ne):

                v = v_value[i][j]
                
                sigma_total = sigma_total+ (v - mean[0][i]) * (v - mean[0][i])
                
            v_sigma = sigma_total / self.Ne
            sigma[0][i] = v_sigma
        total = sigma[0][1] + sigma[0][2] + sigma[0][3] + sigma[0][0] + sigma[0][4] 
        for i in range(5):
            sigma_norm[0][i] = sigma[0][i] / total
        sigma_norm = torch.Tensor(sigma_norm)
        mean = torch.Tensor(mean)
        #print(sigma)
        distibution = sigma_norm
     
        return distibution
    def get_explore_action(self,input):
        input = input.cuda()
        a = self.explorer(input)

        dist = Categorical(a)
        action = dist.sample()  
        log_prob = dist.log_prob(action)   #每种action的概率           #可采取的action
        action = action.cpu()
        return action.detach().numpy(), log_prob,a        
    def simple_get_explore_action(self,distribution,obstacle):
        for i in range (4):
            if obstacle[0][i] == 1:
                distribution[0][i+1] = 0
        max_value = -1
        action = 0
        for i in range(5):
            if distribution[0][i] > max_value:
                action = i
                max_value = distribution[0][i]
        log = 0.5
        a = distribution
        return action,log,a

    def get_KL_divergence(self,distribution,action_prob):
        distribution = distribution.cuda()
        action_prob = action_prob.cuda()
        #print(distribution[0][0].cpu().float(),action_prob[0][0])
        KL = 0
        for i in range(5):
            KL += action_prob[0][i] * math.log(action_prob[0][i]/distribution[0][i],math.e)
        return KL
    def get_JS_divergence(self,distribution,action_prob):
        distribution = distribution.cuda()
        
        action_prob = action_prob.cuda()
        #print(distribution,action_prob)
        M = 1/2 * (distribution + action_prob)
        a = self.get_KL_divergence(distribution,M)
        b = self.get_KL_divergence(action_prob,M)
        return a/2 + b/2
    def get_action(self, input):
        input = input.cuda()
        a = self.actor(input)

        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率
        action = action.cpu()
        return action.detach().numpy(), log_prob,a

    def learn(self, log,s,s_,rewards,re,distibution,action_prob,obstacle,mode):#)
        #使用Critic网络估计状态值
        s = s.cuda()
        s_ = s_.cuda()
        v_total = 0
        v_total_ = 0
        critic_loss = 0
        self.critic_loss = []
        for model in self.ensemble_list:
            vi=model(s)
            vi_=model(s_)
    
            # TD误差
            v_total = v_total + vi
            v_total_ = v_total_ + vi_
            self.critic_loss.append(self.loss(self.gamma*vi_+rewards,vi))
            critic_loss  = critic_loss + self.loss(self.gamma*vi_+rewards,vi)
        self.optimizer.zero_grad()
        for i in range(self.Ne):
            critic_loss = self.critic_loss[i]
            critic_loss.backward()
        self.optimizer.step()
        if mode == 1:
            td = self.gamma * v_total_/self.Ne + rewards - v_total / self.Ne          #计算TD误差
            loss_actor = -log * td.detach() 
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()
        if mode == 0:
            distibution[0][0] = 0
            for i in range (4):
                if obstacle[0][i] == 1:
                    distibution[0][i+1] = 0
            distibution = distibution.cuda()
            loss_explore = self.crossEntropyLoss(distibution,action_prob)
            self.explorer_optim.zero_grad()
            loss_explore.backward()
            self.explorer_optim.step()

        

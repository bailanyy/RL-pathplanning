import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import collections
import random
import math

class Chain():
    def __init__(self):
        self.length = 10
        self.action_num = 3
        self.state_num = 1 
        self.chain = np.zeros((1,self.length))
        self.agent = None
    def update(self):
        action = self.agent.action
        if self.agent.pos == 0:
            if action == 0:
                self.agent.pos = self.agent.pos
            if action == 1:
                self.agent.pos =self.agent.pos
            if action == 2:
                self.agent.pos += 1
        elif self.agent.pos == 49:
            if action == 0:
                self.agent.pos = self.agent.pos
            if action == 1:
                self.agent.pos =self.agent.pos - 1
            if action == 2:
                self.agent.pos = self.agent.pos
        else:
            if action == 0:
                self.agent.pos = self.agent.pos
            if action == 1:
                self.agent.pos =self.agent.pos - 1
            if action == 2:
                self.agent.pos = self.agent.pos + 1
        reward = self.agent.get_reward()
        return reward,self.agent.pos
    def add_agent(self,agent):
        self.agent = agent
class Agent():
    def __init__(self):
        self.pos = 0
        self.action = 0
        self.reached = False
    def get_reward(self):
        return self.pos
    

class Critic(nn.Module):
    '''
    评论家Critic网络
    '''
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(1,8)
        self.l2 = nn.Linear(8,1)
    def forward(self, x):
        #x = torch.reshape(x,(1,1))
        # x = np.array([x])
        #x = torch.Tensor([x])
        x = self.l1(x)
        x = F.relu(x)
        out = self.l2(x)

        return out
    


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



class Explorer(nn.Module):
    def __init__(self):
        super(Explorer, self).__init__()

        self.l1 = nn.Linear(1,8)
        self.l2 = nn.Linear(8,3)
    def forward(self, x):

        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        out = F.softmax(x)
        return out
class EC():
    def __init__(self):
        self.gamma = 0.99
        self.lr_a = 1e-4
        self.lr_c = 1e-3
        self.lr_e = 1e-4
        self.Ne = 5

        self.action_dim = 5       #获取描述行动的数据维度
        
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
        for i in range(self.Ne):
            ensemble_optim = torch.optim.Adam(self.ensemble_list[i].parameters(), lr=self.lr_c)
            self.optim_list.append(ensemble_optim)
        
        
        self.loss = nn.MSELoss().cuda()
        self.crossEntropyLoss = nn.CrossEntropyLoss()
    def get_next(self,input):
        self.all_next = input

    def get_posterior_distribution(self):
        mean = np.zeros((1,3))
        sigma = np.zeros((1,3))
        sigma_norm = np.zeros((1,3))
        posterior = np.zeros((1,3))
        v_value = np.zeros((3,self.Ne))
        for i in range(3):
            v_total = 0
            j=0
            for model in self.ensemble_list:
                s = torch.Tensor([self.all_next[i]]).cuda()
                v = model(s)
                v = v.cpu().detach().float()
                v_value[i][j] = v
                v_total += v
                j += 1
            v_mean = v_total / self.Ne
            mean[0][i] = v_mean
        #print(v_value)
        sigma_total = 0
        for i in range(3):
            v_sigma = 0
            for j in range(self.Ne):

                v = v_value[i][j]
                
                sigma_total = sigma_total+ (v - mean[0][i]) * (v - mean[0][i])
                #print(v,mean[0][i])
            v_sigma = sigma_total / self.Ne
            sigma[0][i] = v_sigma
        total = sigma[0][0] + sigma[0][1] + sigma[0][2] 
        for i in range(3):
            sigma_norm[0][i] = sigma[0][i] / total
        sigma_norm = torch.Tensor(sigma_norm)
        mean = torch.Tensor(mean)
        
        distibution = sigma_norm
     
        return distibution
    def get_explore_action(self,input):
        input = torch.Tensor([input])
        input = input.cuda()
        a = self.explorer(input)

        dist = Categorical(a)
        action = dist.sample()  
        log_prob = dist.log_prob(action)   #每种action的概率           #可采取的action
        action = action.cpu()
        return action.detach().numpy(),a        
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

    def learn(self,s,s_,rewards,distibution,action_prob):#)
        s = torch.Tensor([s])
        s_ = torch.Tensor([s_])
        #使用Critic网络估计状态值
        s = s.cuda()
        s_ = s_.cuda()
      
    
        # v = self.critic(s)
        # v_ = self.critic(s_)
        v_total = 0
        v_total_ = 0
        critic_loss = 0
        for model in self.ensemble_list:
            vi=model(s)
            vi_=model(s_)
    
            # TD误差
            v_total = v_total + vi
            v_total_ = v_total_ + vi_
            self.critic_loss.append(self.loss(self.gamma*vi_+rewards,vi))
            critic_loss  = critic_loss + self.loss(self.gamma*vi_+rewards,vi)
        
        for i in range(self.Ne):
            self.optim_list[i].zero_grad()

        critic_loss.backward()

        for i in range(self.Ne):
            self.optim_list[i].step()
        distibution = distibution.cuda()
        loss_explore = self.crossEntropyLoss(distibution,action_prob)
        self.explorer_optim.zero_grad()
        loss_explore.backward()
        self.explorer_optim.step()
     
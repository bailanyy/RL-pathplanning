'''
Author: yy.yy 729853861@qq.com
Date: 2024-06-20 21:31:36
LastEditors: yygod-sgdie 729853861@qq.com
LastEditTime: 2024-07-02 19:30:08
FilePath: \dissertation_project\dqn\a2c_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: yy.yy 729853861@qq.com
Date: 2024-06-20 21:31:36
LastEditors: yy.yy 729853861@qq.com
LastEditTime: 2024-06-21 21:51:40
FilePath: \dqn\a2c_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

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
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, action_dim)

        self.ln = nn.LayerNorm(100)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = F.softmax(self.fc2(x), dim=-1)

        return out


class Critic(nn.Module):
    '''
    评论家Critic网络
    '''
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, 1)

        self.ln = nn.LayerNorm(100)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out

class Q_Critic(nn.Module):
    '''
    Q评论家Critic网络
    '''
    def __init__(self, state_dim):
        super(Q_Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, 2)
        
        self.ln = nn.LayerNorm(100)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out


class Actor_Critic:
    def __init__(self, env):
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4

        self.env = env
        self.action_dim = self.env.action_space.n             #获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  #获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)   #创建演员网络
        self.critic = Critic(self.state_dim)                  #创建评论家网络
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        self.loss = nn.MSELoss()

    def get_action(self, s):
        input = torch.Tensor(s)

        a = self.actor(input)
        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率

        return action.detach().numpy(), log_prob

    def learn(self, log_prob, s, s_, rew):
        #使用Critic网络估计状态值
        v = self.critic(s)
        v_ = self.critic(s_)

        critic_loss = self.loss(self.gamma * v_ + rew, v)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        td = self.gamma * v_ + rew - v          #计算TD误差
        loss_actor = -log_prob * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()






# Q-based A2C
class Actor_Critic_Q:
    def __init__(self, env):
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4

        self.env = env
        self.action_dim = self.env.action_space.n             #获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  #获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)   #创建演员网络
        self.critic = Q_Critic(self.state_dim)                  #创建评论家网络
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        self.loss = nn.MSELoss()

    def get_action(self, s):
        input = torch.Tensor(s)

        a = self.actor(input)
        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率

        return action.detach().numpy(), log_prob,a

    def learn(self, log_prob, s, s_, rew):
        #使用Critic网络估计状态值

        a,log_p,p = self.get_action(s)
        a_,log_p_,p_ = self.get_action(s_)
        q = self.critic(s)# 当前状态值的动作-状态函数
        q_ = self.critic(s_)
        
        #print(q[1])
        #计算下一个状态的状态值
        
        A = rew + self.gamma * q_[a_]
        critic_loss = self.loss(A, q[a])
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        td = critic_loss        #计算TD误差
        loss_actor = -log_prob * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()









#集成模型


class Actor_Critic_DE:
    def __init__(self, env):
        self.ensemble_weight = 0.1
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4
        self.Ne = 10
        self.env = env
        self.action_dim = self.env.action_space.n             #获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  #获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)   #创建演员网络
        #self.critic = Critic(self.state_dim)                  #创建评论家网络
        self.ensemble_list = []
        self.optim_list = []
        self.critic_loss = []
        for i in range(self.Ne):
            tmp_critic = Critic(self.state_dim)
            tmp_critic.apply(init_weights)
            self.ensemble_list.append(tmp_critic)
        for i in range(self.Ne):
            ensemble_optim = torch.optim.Adam(self.ensemble_list[i].parameters(), lr=self.lr_c)
            self.optim_list.append(ensemble_optim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        
        self.loss = nn.MSELoss()

        self.miu = 0
        self.sigma = 0
        self.sigma_norm = 0
        self.distibution = 0
        self.v_1 = []

    def get_action(self, s):
        input = torch.Tensor(s)

        a = self.actor(input)
        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率

        return action.detach().numpy(), log_prob

    def learn(self, log_prob, s, s_, rew):
        #应用网络的输出
        v_total = 0
        v_total_ = 0
        critic_loss = 0
        for model in self.ensemble_list:
            vi=model(s)
            vi_=model(s_)
            self.v_1.append(vi_)
            # TD误差
            v_total = v_total + vi
            v_total_ = v_total_ + vi_
            self.critic_loss.append(self.loss(self.gamma*vi_+rew,vi))
            critic_loss  = critic_loss + self.loss(self.gamma*vi_+rew,vi)
        for i in range(self.Ne):
            self.sigma = self.sigma + pow(self.v_1[i] - self.miu,2)/self.Ne

        
        for i in range(self.Ne):
            self.optim_list[i].zero_grad()

        critic_loss.backward()

        for i in range(self.Ne):
            self.optim_list[i].step()
        


        td = self.gamma * v_total_/self.Ne + rew - v_total / self.Ne         #计算TD误差
        #td = critic_loss / self.Ne
        loss_actor = -log_prob * td.detach()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()


import math
class Actor_Critic_QDE:
    def __init__(self, env):
        self.gamma = 0.99
        self.lr_a = 3e-4
        self.lr_c = 5e-4
        self.Ne = 10
        self.env = env
        self.action_dim = self.env.action_space.n             #获取描述行动的数据维度
        self.state_dim = self.env.observation_space.shape[0]  #获取描述环境的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)   #创建演员网络
        self.ensemble_list = []
        self.optim_list = []
        self.critic_loss = torch.zeros(1,2)
        self.KL = 0
        self.wkl = 1e-3
        for i in range(self.Ne):
            tmp_critic = Q_Critic(self.state_dim)
            tmp_critic.apply(init_weights)
            self.ensemble_list.append(tmp_critic)
        for i in range(self.Ne):
            ensemble_optim = torch.optim.Adam(self.ensemble_list[i].parameters(), lr=self.lr_c)
            self.optim_list.append(ensemble_optim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        
        self.loss = nn.MSELoss()

        self.miu = []
        self.sigma = []
        self.sigma_norm = []
        self.distibution = []

    def get_action(self, s):
        input = torch.Tensor(s)

        a = self.actor(input)
        dist = Categorical(a)
        action = dist.sample()             #可采取的action
        log_prob = dist.log_prob(action)   #每种action的概率

        return action.detach().numpy(), log_prob,a

    def learn(self, log_prob, s, s_, rew):
        #应用网络的输出
        a,log_p,p = self.get_action(s)
        a_,log_p_,p_ = self.get_action(s_)
        critic_loss = 0
        A_total = 0
        qa = 0
        for model in self.ensemble_list:
            q=model(s)
            q_=model(s_)
            # TD误差
            
            A = rew + self.gamma * q_[a_]
   
            critic_loss  = critic_loss + (A - q[a])#+ self.loss(A,q[a])
            A_total = A_total + A
            qa = qa + q[a]
        miu0 = 0
        miu1 = 0
        for i in range(self.Ne):
            miu0 = miu0 + self.ensemble_list[i](s_)[0]
            miu1 = miu1 + self.ensemble_list[i](s_)[1]
        self.miu = [miu0/self.Ne,miu1/self.Ne]
        sigma0 = 0
        sigma1 = 0
        for i in range(self.Ne):
            sigma0 = sigma0 + pow(self.ensemble_list[i](s_)[0] - miu0,2)
            sigma1 = sigma1 + pow(self.ensemble_list[i](s_)[1] - miu1,2)
        self.sigma = [sigma0/self.Ne,sigma1/self.Ne]
        self.sigma_norm = [self.sigma[0]/sigma0,self.sigma[1]/sigma1]
        input = torch.Tensor(self.sigma_norm)
        self.distibution = F.softmax(input)
        self.KL = p[0] * math.log(p[0]/self.distibution[0],math.e) + p[1] * math.log(p[1]/self.distibution[1],math.e)
        #print(self.KL)
        for i in range(self.Ne):
            self.optim_list[i].zero_grad()
        
        critic_loss.backward()

        for i in range(self.Ne):
            self.optim_list[i].step()
        

        #td = A_total / self.Ne  - qa / self.Ne
        td = critic_loss/self.Ne
        
        loss_actor = -log_prob * td.detach() + self.wkl * self.KL
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
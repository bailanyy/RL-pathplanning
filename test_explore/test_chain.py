'''
Author: yygod-sgdie 729853861@qq.com
Date: 2024-07-04 22:24:10
LastEditors: yygod-sgdie 729853861@qq.com
LastEditTime: 2024-07-13 15:46:19
FilePath: \dissertation_project\test_explore\test_chain.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from mdp_chain import Chain
from mdp_chain import Agent
from mdp_chain import EC
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_next_state(pos):
    next = []
    if pos == 0:
        next = [pos,pos,pos+1]
    elif pos == 49:
        next = [pos,pos-1,pos]
    else:
        next = [pos,pos-1,pos+1]
    return next



chain = Chain()
agent = Agent()
model = EC()
chain.add_agent(agent)
for k in range (20):
    chain.agent.pos = 0
    for e in range(100):
        pos = agent.pos
        next = get_next_state(pos)
        model.get_next(next)

        d = model.get_prior_distribution() #得到先验分布
        chain.agent.action,action_prob = model.get_explore_action(pos)
        chain.agent.action = np.random.randint(0,3)
        action_prob = torch.reshape(action_prob,(1,3))
        reward,state = chain.update()
        d_ = model.get_prior_distribution() #后验 
        model.learn(pos,state,reward,d,d_,action_prob)
        d_ = model.get_prior_distribution() #后验 
        #print(d,chain.agent.pos,next)
        
        

explorer_plot = []
random_plot = []
chain.agent.pos = 0
for i in range(100):
    pos = agent.pos
    chain.agent.action,action_prob = model.get_explore_action(pos)
    reward,state = chain.update()
    explorer_plot.append(pos)
chain.agent.pos = 0
for i in range(100):
    pos = agent.pos
    chain.agent.action = np.random.randint(0,3)
    reward,state = chain.update()
    random_plot.append(pos)

print(random_plot)
plt.plot(explorer_plot)
plt.plot(random_plot)
plt.show()

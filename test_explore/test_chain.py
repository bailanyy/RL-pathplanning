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
for k in range (10):
    chain.agent.pos = 0
    for e in range(100):
        pos = agent.pos
        next = get_next_state(pos)
        model.get_next(next)
        d = model.get_posterior_distribution()
        chain.agent.action,action_prob = model.get_explore_action(pos)
        action_prob = torch.reshape(action_prob,(1,3))
        
        reward,state = chain.update()
        model.learn(pos,state,reward,d,action_prob)
    
    print(k,chain.agent.pos)

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

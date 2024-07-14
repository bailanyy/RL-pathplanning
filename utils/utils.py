import numpy as np
from env import Agent
import sys
sys.path.append(r'C:\workspace\dissertation_project\a2c')
from a2c import A2C_DE
class method:
    def __init__(self):
        u = 0
    def add_agent(num,env):
        for i in range(num):
            a = Agent()
            env.add_agent(a)

    def add_guide(num,env):
        # id 0
        x0 = np.loadtxt("guidex.txt")
        y0 = np.loadtxt("guidey.txt")
        # id 1
        x1 = np.loadtxt("guidex1.txt")
        y1 = np.loadtxt("guidey1.txt")
        
        env.global_x_list.append(x0)
        env.global_y_list.append(y0)

        env.global_x_list.append(x1)
        env.global_y_list.append(y1)
    def initial_agent(num,env):
        for i in range(num):
            env.agent_list[i].reached = False
            env.agent_list[i].global_count = 1
    def multi_ob(map,env,num):
        for i in range(num):
            env.agent_list[i].obversation(map)
    def agent_get_guide(num,env):
        for i in range(num):
            env.agent_list[i].get_guide_arrary(env.global_x_list[i],env.global_y_list[i])
    def get_model_list(num):
        model_list = []
        for i in range(num):
            a = A2C_DE()
            model_list.append(a)
        return model_list
    def get_initial_ob(env):
        state = []
        for i in range(env.agent_num):
            ob = env.agent_list[i].transVOF2tensor()
            state.append(ob)
        return state    
    def get_multi_goal(env):
        
        env.agent_list[0].goal[0][0] = 39
        env.agent_list[0].goal[0][1] = 39
        env.agent_list[1].goal[0][0] = 39
        env.agent_list[1].goal[0][1] = 0
'''
Author: yy.yy 729853861@qq.com
Date: 2024-06-24 16:18:19
LastEditors: yygod-sgdie 729853861@qq.com
LastEditTime: 2024-07-14 21:37:59
FilePath: \dqnc:\workspace\dissertation_project\env\test_env.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
from env import Env
from env import Agent
from global_guide import astar
from global_guide import Map
sys.path.append(r'C:\workspace\dissertation_project\utils')
from utils import method
import time

sys.path.append(r'C:\workspace\dissertation_project\a2c')
from a2c import Actor_Critic
from a2c import A2C_DE

from a2c import ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
env = Env()
model = A2C_DE()
#agent = Agent()
a1 = Agent()
a2 = Agent()
m1 = A2C_DE()
m2 = A2C_DE()
model_list = []
model_list.append(m1)
model_list.append(m2)


env.grid_map = np.loadtxt('map.txt')

#method.add_agent(2,env)  # 添加了多个智能体
env.agent_list.append(a1)
env.agent_list.append(a2)
env.agent_list[0].action = 0
env.agent_list[1].action = 0
env.reset()

#env.generate_multi_global_guidence()
#env.update()
maze = env.grid_map

# 需要为每个智能体添加全局导航

method.add_guide(2,env)




import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    WALL_COLOR = (255, 255, 255)
    ROUTE_COLOR = (0, 0, 0)
    AGENT_COLOR = (255,0,0)
    GUIDE_COLOR = (0,255,0)
    maze = env.grid_map
    
    success = 0
    for episode in range(300):
        env.done = False

        method.initial_agent(2,env)
        env.reset()     
        print(env.agent_list[1].position)
        method.multi_ob(maze,env,2)
        
        max_step = 0
        env.get_multi_ob()
        ep_r = 0
        state = method.get_initial_ob(env)
        x1 = np.loadtxt("guidex.txt")
        y1 = np.loadtxt("guidey.txt")
        x2 = np.loadtxt("guidex1.txt")
        y2 = np.loadtxt("guidey1.txt")
        env.agent_list[0].get_guide_arrary(x1,y1)
        
        env.agent_list[1].get_guide_arrary(x2,y2)
        env.generate_multi_global_guidence()
        method.get_multi_goal(env)
        mode = 0 # explore 0 actor 1
        
        while True:
            maze = env.grid_map
            
            env.generate_multi_global_guidence()     
                  
            # 动画部分
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            for i in range(40):
                for j in range(40):
                    
                    if env.global_guide_map[i][j] == 1: #green
                        pygame.draw.rect(screen, GUIDE_COLOR, pygame.Rect(j * 10, i * 10, 10, 10))

                    else:
                        if maze[i][j] == 0: # white
                            pygame.draw.rect(screen, WALL_COLOR, pygame.Rect(j * 10, i * 10, 10, 10))
                        
                        if maze[i][j] == 1: #black
                            pygame.draw.rect(screen, ROUTE_COLOR, pygame.Rect(j * 10, i * 10, 10, 10))
                    if maze[i][j] == -1: # red
                            pygame.draw.rect(screen, AGENT_COLOR, pygame.Rect(j * 10, i * 10, 10, 10))
            for i in range(env.agent_num):
                next = env.agent_list[i].get_all_state(env.grid_map)
                model_list[i].get_next(next)
                x = model_list[i].get_posterior_distribution()
           
                obstacle = env.agent_list[i].get_four_direction(env.grid_map)
            # take action 
                #p = 0.2 - 0.05*episode
                p = 0
                if np.random.rand() > p:
                    if env.agent_list[i].reached != True:
                        env.agent_list[i].action,env.agent_list[i].log_prob,env.agent_list[i].action_prob= model_list[i].get_action(state[i])
                    mode = 1
                else:
                    env.agent_list[i].action,env.agent_list[i].log_prob,env.agent_list[i].action_prob= model_list[i].get_explore_action(state[i])
             
                    mode = 0
         
            reward,re,state_,done = env.update() # 更新下一步的地图,返回一个reward

            # 挨个训练
            for i in range(env.agent_num):
                rew = reward[i]
                s_ = state[i]
                r = re[i]
                ep_r = ep_r + rew
                
                if env.agent_list[i].reached != True:
                    model_list[i].learn(env.agent_list[i].log_prob,state[i],s_,rew,r,x,env.agent_list[i].action_prob,obstacle,mode)
            
            if max_step > 300:
                    done = True
            if done == True:
                if env.agent_list[0].reached == True and env.agent_list[1].reached == True:
                    success = success + 1
                break
            max_step = max_step + 1
            state = state_
            pygame.display.update()
        print(f"episode:{episode} ep_r:{ep_r} success:{success}")
if __name__ == '__main__':
    main()
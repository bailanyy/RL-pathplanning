'''
Author: yy.yy 729853861@qq.com
Date: 2024-06-24 16:18:19
LastEditors: yygod-sgdie 729853861@qq.com
LastEditTime: 2024-07-14 20:44:30
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
agent = Agent()

#env.generate_map()
env.grid_map = np.loadtxt('map.txt')
env.add_agent(agent)
#method.add_agent(2,env,agent)  # 添加了多个智能体
env.agent_list[0].action = 0
env.update()
maze = env.grid_map

# 需要为每个智能体添加全局导航
env.global_guide_map = np.loadtxt('guide1.txt')
resultx = np.loadtxt("guidex1.txt")
resulty = np.loadtxt('guidey1.txt')

cross_list = []
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
        env.agent_list[0].reached = False
        env.agent_list[0].global_count = 1
        env.reset()
        
 
        resultx = np.loadtxt("guidex1.txt")
        resulty = np.loadtxt('guidey1.txt')
        env.generate_global_guidence(resultx,resulty)
        env.agent_list[0].obversation(maze)
        max_step = 0
        
        s = env.agent_list[0].transVOF2tensor()
        ep_r = 0
        kl_total = 0
        env.agent_list[0].get_guide_arrary(resultx,resulty)
        mode = 0 # explore 0 actor 1
        explore_map =np.loadtxt('guide.txt')
        rate = 0
        while True:
            maze = env.grid_map
            
            
            env.generate_global_guidence(env.agent_list[0].guidex,env.agent_list[0].guidey)
            #print(env.agent_list[0].guidex,env.agent_list[0].guidey)
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
                        if maze[i][j] == -1: # red
                            pygame.draw.rect(screen, AGENT_COLOR, pygame.Rect(j * 10, i * 10, 10, 10))
                        if maze[i][j] == 1: #black
                            pygame.draw.rect(screen, ROUTE_COLOR, pygame.Rect(j * 10, i * 10, 10, 10))
            next = env.agent_list[0].get_all_state(env.grid_map)
            model.get_next(next)
            x = model.get_posterior_distribution()
           
            obstacle = env.agent_list[0].get_four_direction(env.grid_map)
            #print(obstacle)
            # take action 
            p = 0#0.2 - 0.05*episode
            if np.random.rand() > p:
            #env.agent_list[0].action,log_prob,a= model.get_action(s)
         
                env.agent_list[0].action,log_prob,a= model.get_action(s)
                mode = 1
            else:
                env.agent_list[0].action,log_prob,a= model.get_explore_action(s)
            #env.agent_list[0].action =  np.random.randint(0,5)
                mode = 0
            #env.agent_list[0].action,log_prob,a= model.simple_get_explore_action(model.get_posterior_distribution(),obstacle)
            reward,re,state,done = env.update() # 更新下一步的地图,返回一个reward
            #print(explore_map[env.agent_list[0].position[0][0].astype('int64')][env.agent_list[0].position[0][1].astype('int64')],env.agent_list[0].position[0][0],env.agent_list[0].position[0][1])
            if explore_map[env.agent_list[0].position[0][0].astype('int64')][env.agent_list[0].position[0][1].astype('int64')] == 0:
                rate += 1
                explore_map[env.agent_list[0].position[0][0].astype('int64')][env.agent_list[0].position[0][1].astype('int64')] = 1
            rew = reward[0]
            s_ = state[0]
            r = re[0]
            ep_r = ep_r + rew
            if max_step > 300:
                done = True
            if done == True:
                if env.agent_list[0].reached == True:
                    success = success + 1
                break

            model.learn(log_prob,s,s_,rew,r,x,a,obstacle,mode)
            
            #print(x,rate)
            #print(f"episode:{max_step} ep_r:{kl} action{env.agent_list[0].action}")
            max_step = max_step + 1
            s = s_
            #print(ep_r)
            pygame.display.update()
        print(f"episode:{episode} ep_r:{ep_r} success:{success} kl:{kl_total}")
    # plt.plot(cross_list)
    # plt.show()
if __name__ == '__main__':
    main()
'''
Author: yy.yy 729853861@qq.com
Date: 2024-06-24 16:18:19
LastEditors: yygod-sgdie 729853861@qq.com
LastEditTime: 2024-06-30 14:31:40
FilePath: \dqnc:\workspace\dissertation_project\env\test_env.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from env import Env
from env import Agent
from global_guide import astar
from global_guide import Map
import time
import sys
sys.path.append(r'C:\workspace\dissertation_project\a2c')
from a2c import Actor_Critic
from a2c import ReplayBuffer
import numpy as np
env = Env()
model = Actor_Critic()
agent = Agent()

#env.generate_map()
env.grid_map = np.loadtxt('map.txt')
env.add_agent(agent)
#np.savetxt('map.txt',env.grid_map)
env.agent_list[0].action = 0
env.update()
maze = env.grid_map
# map = Map(maze,0,0,39,39)
# resultx,resulty = astar(map)   
# resultx.reverse()  #得到了全局的路径规划
# resulty.reverse()  #得到了全局的路径规划
# env.generate_global_guidence(resultx,resulty)
# np.savetxt('guide.txt',env.global_guide_map)
# np.savetxt('guidex.txt',resultx)
# np.savetxt('guidey.txt',resulty)
env.global_guide_map = np.loadtxt('guide.txt')
resultx = np.loadtxt("guidex.txt")
resulty = np.loadtxt('guidey.txt')
print(env.global_guide_map)
print(env.grid_map)
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
    for episode in range(3000):
        env.done = False
        env.agent_list[0].reached = False
        env.agent_list[0].global_count = 1
        env.reset()
        
        #env.global_guide_map = np.loadtxt('guide.txt')
        resultx = np.loadtxt("guidex.txt")
        resulty = np.loadtxt('guidey.txt')
        env.generate_global_guidence(resultx,resulty)
        env.agent_list[0].obversation(maze)
        max_step = 0
        
        s = env.agent_list[0].transVOF2tensor()
        ep_r = 0
        env.agent_list[0].get_guide_arrary(resultx,resulty)
        
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
                    
            # take action 
            env.agent_list[0].action,log_prob= model.get_action(s)

            reward,state,done = env.update() # 更新下一步的地图,返回一个reward
            rew = reward[0]
            s_ = state[0]
            ep_r = ep_r + rew
            if max_step > 256:
                done = True
            if done == True:
                if env.agent_list[0].reached == True:
                    success = success + 1
                    print(rew)
                break
            model.learn(log_prob,s,s_,rew)
            
            max_step = max_step + 1
            s = s_
            #print(ep_r)
            pygame.display.update()
        print(f"episode:{episode} ep_r:{ep_r} success:{success}")
    print(success)
if __name__ == '__main__':
    main()
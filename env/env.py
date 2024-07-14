'''
Author: yygod-sgdie 729853861@qq.com
Date: 2024-06-24 15:01:53
LastEditors: yygod-sgdie 729853861@qq.com
LastEditTime: 2024-07-14 21:36:19
FilePath: \dissertation_project\env\env.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
import torchvision.transforms as transforms
import math
class Env:
    def __init__(self):
        self.W = 40
        self.H = 40
        
        self.grid_map = np.zeros((self.W,self.H))
        self.global_guide_map = np.zeros((self.W,self.H))
        self.global_guide_list = []
        self.global_x_list = []
        self.global_y_list = []
        self.agent_num = 2
        self.done_count = self.agent_num
        self.agent_list = []
        self.reward_list = []
        self.explore_reward_list = []
        self.observe_list = []
        self.state_list = []
        self.obstacle_num = 100
        self.eposide_end = False
        self.done = False
    def generate_map(self):
        rand_obstacle = np.random.randint(0,40,[2,self.obstacle_num])
        for i in range(self.obstacle_num):
            self.grid_map[rand_obstacle[0][i].astype('int64')][rand_obstacle[1][i].astype('int64')] = 1
    def generate_global_guidence(self,guidex,guidey):
        self.global_guide_map = np.zeros((self.W,self.H))
        for i in range(len(guidex)):
            self.global_guide_map[guidex[i].astype('int64')][guidey[i].astype('int64')] = 1
    def generate_multi_global_guidence(self):
        self.global_guide_map = np.zeros((self.W,self.H))
        self.global_guide_list = []
        
        for i in range(self.agent_num):
            gmap = np.zeros((self.W,self.H))
            for j in range(len(self.agent_list[i].guidex)):
                
                gmap[self.agent_list[i].guidex[j].astype('int64')][self.agent_list[i].guidey[j].astype('int64')] = 1
                self.global_guide_map[self.agent_list[i].guidex[j].astype('int64')][self.agent_list[i].guidey[j].astype('int64')] = 1
            self.global_guide_list.append(gmap)

    def add_agent(self,agent):
        for i in range(self.agent_num):
            self.agent_list.append(agent)
        
            self.grid_map[agent.position[0][0].astype('int64')][agent.position[0][1].astype('int64')] = -1
    def get_multi_ob(self):
        for i in range(self.agent_num):
            s = self.agent_list[i].transVOF2tensor()
            self.state_list.append(s)
    def update(self):
        self.reward_list = []
        self.observe_list = []
        self.explore_reward_list = []
        for i in range(self.agent_num):
            if self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] == 1:
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = 1
            else:
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = 0
            
            # 得到动作
            action = self.agent_list[i].action
            if self.agent_list[i].reached == True:
                #self.done = True
                self.agent_list[i].action = 0

            if action == 0:
                self.agent_list[i].position = self.agent_list[i].position
            if action == 1:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] - 1
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] + 0
            if action == 2:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] + 0
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] - 1
            if action == 3:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] + 1
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] - 0
            if action == 4:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] + 0
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] + 1
          
            
            if self.agent_list[i].position[0][0] < 0 or self.agent_list[i].position[0][0] > 39 or self.agent_list[i].position[0][1] <0 or self.agent_list[i].position[0][1]>39:
                self.agent_list[i].out_bounded_flag  = True  # 越界
                self.reset_position(action,self.agent_list[i])
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = -1
            elif self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] == 1:
                self.agent_list[i].collision_flag = True  # 标志碰撞信号 碰撞障
                self.reset_position(action,self.agent_list[i])
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = -1
            else:
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = -1
            reward,re = self.agent_list[i].get_reward(self.global_guide_list[i])  # 错了
            self.reward_list.append(reward)  #在这个函数置到达标志位
            self.explore_reward_list.append(re)
            # 不对，只对一个智能体进行判断，在多智能体时需要更改进行更复杂的判断
            
            self.agent_list[i].obversation(self.grid_map)
            ob = self.agent_list[i].transVOF2tensor()
            self.observe_list.append(ob)
        if self.agent_list[0].reached == True and self.agent_list[1].reached == True:
            self.done = True
        return self.reward_list,self.explore_reward_list,self.observe_list,self.done
    def reset(self):
        for i in range(self.agent_num):
            self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = 0
            if i == 0:
                self.agent_list[0].position = np.zeros((1,2))
            if i == 1:
                p = np.zeros((1,2))
                p[0][0] = 0
                p[0][1] = 39
                self.agent_list[1].position = p
    
    def reset_position(self,action,agent):
        if action == 0:
            agent.position = agent.position
        if action == 1:
            agent.position[0][0] =agent.position[0][0] + 1
            agent.position[0][1] = agent.position[0][1] + 0
        if action == 2:
            agent.position[0][0] = agent.position[0][0] + 0
            agent.position[0][1] = agent.position[0][1]+ 1
        if action == 3:
            agent.position[0][0] = agent.position[0][0] - 1
            agent.position[0][1] = agent.position[0][1] + 0
        if action == 4:
            agent.position[0][0] = agent.position[0][0] - 0
            agent.position[0][1] = agent.position[0][1] - 1



class Agent:
    def __init__(self):
        self.action_num = 9
        self.action_space = np.array([0,1,2,3,4,5,6,7,8])
        self.init_position = np.zeros((1,2))
        self.position = np.zeros((1,2))
        self.last_position = np.zeros((1,2))
        self.vof_env = np.zeros((15,15))
        self.vof_state = np.zeros((15,15))
        self.vof_guidence = np.zeros((15,15))
        self.explore_map = np.zeros((40,40))
        self.vof = [self.vof_env,self.vof_state,self.vof_guidence]
        self.action = -1
        self.log_prob = None
        self.action_prob = None
        self.goal = np.zeros((1,2))
        self.goal[0][0] = 39
        self.goal[0][1] = 39
        self.collision_flag = False
        self.out_bounded_flag = False
        self.reached = False 
        self.global_count = 0
        self.guidex = None
        self.guidey = None
    def get_explore_map(self,map):
        self.explore_map = map
    def get_all_state(self,grid_map):
        next_state = []
        for i in range(5):
            env = np.zeros((15,15))
            state = np.zeros((15,15))
            guide = np.zeros((15,15))
            if i == 0:
                start_i = self.position[0][0].astype('int64') - 7
                start_j = self.position[0][1].astype('int64') - 7
            if i == 1:
                start_i = self.position[0][0].astype('int64') - 7 - 1
                start_j = self.position[0][1].astype('int64') - 7
            if i == 2:
                start_i = self.position[0][0].astype('int64') - 7
                start_j = self.position[0][1].astype('int64') - 1 - 7
            if i == 3:
                start_i = self.position[0][0].astype('int64') - 7 + 1
                start_j = self.position[0][1].astype('int64') - 7
            if i == 4:
                start_i = self.position[0][0].astype('int64') - 7
                start_j = self.position[0][1].astype('int64') - 7 + 1
            for i in range(15):
                for j in range(15):
                    if self.out_boundness(start_i + i,start_j + j) == True:
                        env[i][j] = 1
                    else:
                        if grid_map[start_i + i][start_j + j] == 1:
                            env[i][j] = 1
                                    
                        if grid_map[start_i + i][start_j + j] == -1:
                            state[i][j] = 1
                        if grid_map[start_i + i][start_j + j] == 2:
                            guide[i][j] = 1
            env_tensor = torch.tensor(env.astype('float32'))
            state_tensor = torch.tensor(state.astype('float32'))
            guidence_tensor = torch.tensor(guide.astype('float32'))
            input_tensor = torch.stack((env_tensor, state_tensor,guidence_tensor), dim=0)
            next_state.append(input_tensor)
        return next_state
            


    def obversation(self,grid_map): # 越界表示为障碍物
        # map size is 40*40
        start_i = self.position[0][0].astype('int64') - 7
        start_j = self.position[0][1].astype('int64') - 7

        for i in range(15):
            for j in range(15):
                if self.out_boundness(start_i + i,start_j + j) == True:
                    self.vof_env[i][j] = 1
                else:
                    if grid_map[start_i + i][start_j + j] == 1:
                        self.vof_env[i][j] = 1
                                
                    if grid_map[start_i + i][start_j + j] == -1:
                        self.vof_state[7][7] = 0
                        self.vof_state[i][j] = 1
                    if grid_map[start_i + i][start_j + j] == 2:
                        self.vof_guidence[i][j] = 1
    def out_boundness(self,x,y):
        #map size 40*40
        #vof size 15*15
        if x < 0 or x > 39 or y < 0 or y > 39:
            return True
        else:
            return False
    def get_guide_arrary(self,guidex,guidey):
        self.guidex = guidex
        self.guidey = guidey
    def transVOF2tensor(self):
        env_tensor = torch.tensor(self.vof_env.astype('float32'))
        state_tensor = torch.tensor(self.vof_state.astype('float32'))
        guidence_tensor = torch.tensor(self.vof_guidence.astype('float32'))
        input_tensor = torch.stack((env_tensor, state_tensor,guidence_tensor), dim=0)
        return input_tensor
    def get_goal(self,goal):
        self.goal[0][0] = goal[0][0]
        self.goal[0][1] = goal[0][1]
    def get_reward(self,guide):
        r1 = 0.01 * -1
        r_e = 0
        reward = 0
        goal_rew = 50
        if self.collision_flag == True or self.out_bounded_flag == True:
            r2 = -0.1
            r_e = -1
            r3 =0  
        else :
            r2 = 0
            r_e = 0
            # 符合全局指导路线
            if guide[self.position[0][0].astype('int64')][self.position[0][1].astype('int64')] == 1:
                
                
                r3 = 0.1
                reward = reward   + 0.1 * self.global_count
                
                self.global_count = 1
                skip = 0
                # 删除前面一段的全局指导
                for i in range (len(self.guidex)):
                    if self.guidex[i] == self.position[0][0] and self.guidey[i] == self.position[0][1]:
                        skip = i
                        
                        #print(skip,self.guidex[i],self.position[0][0])
                        break
                delte_range = np.arange(skip+1)
                #print(delte_range)
                if len(self.guidex) > 1:
                    self.guidex = np.delete(self.guidex,delte_range)
                    self.guidey = np.delete(self.guidey,delte_range)
                guide[self.position[0][0].astype('int64')][self.position[0][1].astype('int64')] = 0
            else:
                self.global_count += 1   
            if self.position[0][0] == self.goal[0][0] and self.position[0][1] == self.goal[0][1]:
                reward = reward + goal_rew
                
                self.reached = True

      
        reward = reward + r1+  r2 
        self.collision_flag = False
        self.out_bounded_flag = False
        return reward,r_e
    def get_four_direction(self,map):
        # 1 2 3 4 
        obstacale_vector = np.zeros((1,4))
        startx = self.position[0][0].astype('int')
        starty = self.position[0][1].astype('int')
        # action 1
        obstacale_vector[0][0] = self.judge(startx-1,starty,map)
        obstacale_vector[0][1] = self.judge(startx,starty-1,map)
        obstacale_vector[0][2] = self.judge(startx+1,starty,map)
        obstacale_vector[0][3] = self.judge(startx,starty+1,map)
        return obstacale_vector
    def judge(self,startx,starty,map):
        if self.out_boundness(startx,starty)==True or map[startx][starty] != 0:
            return 1
        else:
            return 0
            









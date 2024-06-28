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
        self.agent_num = 1
        self.agent_list = []
        self.reward_list = []
        self.observe_list = []
        self.obstacle_num = 100
        self.eposide_end = False
        self.done = False
    def generate_map(self):
        rand_obstacle = np.random.randint(0,40,[2,self.obstacle_num])
        for i in range(self.obstacle_num):
            self.grid_map[rand_obstacle[0][i]][rand_obstacle[1][i]] = 1
    def generate_global_guidence(self,guidex,guidey):
        for i in range(len(guidex)):
            self.global_guide_map[guidex[i]][guidey[i]] = 1
    def add_agent(self,agent):
        for i in range(self.agent_num):
            self.agent_list.append(agent)
            
            self.grid_map[agent.position[0][0].astype('int64')][agent.position[0][1].astype('int64')] = -1
    def update(self):
        self.reward_list = []
        self.observe_list = []
        for i in range(self.agent_num):
            if self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] == 1:
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = 1
            else:
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = 0
            action = self.agent_list[i].action

            if action == 0:
                self.agent_list[i].position = self.agent_list[i].position
            if action == 1:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] - 1
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] + 0
            if action == 2:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] - 1
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1]- 1
            if action == 3:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] - 0
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] - 1
            if action == 4:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] + 1
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] - 1
            if action == 5:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] + 1
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] + 0
            if action == 6:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] + 1
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] + 1
            if action == 7:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] + 0
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] + 1
            if action == 8:
                self.agent_list[i].position[0][0] = self.agent_list[i].position[0][0] - 1
                self.agent_list[i].position[0][1] = self.agent_list[i].position[0][1] + 1
            
            if self.agent_list[i].position[0][0] < 0 or self.agent_list[i].position[0][0] > 39 or self.agent_list[i].position[0][1] <0 or self.agent_list[i].position[0][1]>39:
                self.agent_list[i].out_bounded_flag  = True  # 越界
                #self.reset()
                self.reset_position(action,self.agent_list[i])
                #self.agent_list[i].position = self.agent_list[i].last_position
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = -1
            elif self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] == 1:
                self.agent_list[i].collision_flag = True  # 标志碰撞信号 碰撞障碍物
                #self.reset()
                self.reset_position(action,self.agent_list[i])
                #self.agent_list[i].position = self.agent_list[i].last_position
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = -1
            else:
                self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = -1
            reward = self.agent_list[i].get_reward(self.global_guide_map)
            self.reward_list.append(reward)  #在这个函数置到达标志位
            # 不对，只对一个智能体进行判断，在多智能体时需要更改进行更复杂的判断
            if self.agent_list[i].reached == True:
                self.done = True

            self.agent_list[i].obversation(self.grid_map)
            ob = self.agent_list[i].transVOF2tensor()
            self.observe_list.append(ob)
        return self.reward_list,self.observe_list,self.done
    def reset(self):
        for i in range(self.agent_num):
            self.grid_map[self.agent_list[i].position[0][0].astype('int64')][self.agent_list[i].position[0][1].astype('int64')] = 0
            self.agent_list[i].position = self.agent_list[i].init_position
            self.agent_list[i].init_position = np.zeros((1,2))
            
    def reset_position(self,action,agent):
        if action == 0:
            agent.position = agent.position
        if action == 1:
            agent.position[0][0] =agent.position[0][0] + 1
            agent.position[0][1] = agent.position[0][1] + 0
        if action == 2:
            agent.position[0][0] = agent.position[0][0] + 1
            agent.position[0][1] = agent.position[0][1]+ 1
        if action == 3:
            agent.position[0][0] = agent.position[0][0] + 0
            agent.position[0][1] = agent.position[0][1] + 1
        if action == 4:
            agent.position[0][0] = agent.position[0][0] - 1
            agent.position[0][1] = agent.position[0][1] + 1
        if action == 5:
            agent.position[0][0] = agent.position[0][0] - 1
            agent.position[0][1] = agent.position[0][1] + 0
        if action == 6:
            agent.position[0][0] = agent.position[0][0] - 1
            agent.position[0][1] = agent.position[0][1] - 1
        if action == 7:
            agent.position[0][0] = agent.position[0][0] + 0
            agent.position[0][1] = agent.position[0][1] - 1
        if action == 8:
            agent.position[0][0] = agent.position[0][0] + 1
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
        self.vof = [self.vof_env,self.vof_state,self.vof_guidence]
        self.action = -1
        self.goal = np.zeros((1,2))
        self.goal[0][0] = 39
        self.goal[0][1] = 39
        self.dis = math.sqrt(39*39+39*39)
        self.collision_flag = False
        self.count = 0
        self.out_bounded_flag = False
        self.reached = False 
        self.global_count = 0
       
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
        r1 = 0.1 * -1
        reward = 0
        goal_rew = 50
        if self.collision_flag == True or self.out_bounded_flag == True:
            r2 = -0.1
            r3 =0
        else :
            r2 = 0
            if guide[self.position[0][0].astype('int64')][self.position[0][1].astype('int64')] == 1:
                r3 = 0.1
                self.global_count = self.global_count + 1
                reward = reward + 0.1 * self.global_count
                #guide[self.position[0][0].astype('int64')][self.position[0][1].astype('int64')] =0
            else:
                r3 = 0
            if self.position[0][0] == self.goal[0][0] and self.position[0][1] == self.goal[0][1]:
                reward = reward + goal_rew
                print('success',reward)
                self.reached = True
            if self.action == 0:
                reward = reward - 0.5
            # 代理需要一定的方向指引
        distance = math.sqrt(pow(self.position[0][0]-self.goal[0][0],2) + pow(self.position[0][0]-self.goal[0][0],2))
        rd = (self.dis - distance)*0.01
        reward = reward + r1+  r2 + rd
        self.collision_flag = False
        self.out_bounded_flag = False
        return reward







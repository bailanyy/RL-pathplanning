'''
Author: yygod-sgdie 729853861@qq.com
Date: 2024-07-13 21:32:52
LastEditors: yygod-sgdie 729853861@qq.com
LastEditTime: 2024-07-14 13:32:58
FilePath: \dissertation_project\env\generate_multi_guidence.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from env import Env
from env import Agent
from global_guide import astar
from global_guide import Map
import numpy as np
env = Env()


env.grid_map = np.loadtxt('map.txt')
#env.add_agent(agent)

#np.savetxt('map.txt',env.grid_map)
maze = env.grid_map
map = Map(maze,0,39,39,0)
resultx1,resulty1 = astar(map)   
resultx1.reverse()  #得到了全局的路径规划
resulty1.reverse()  #得到了全局的路径规划
#env.generate_global_guidence(resultx1,resulty1)
#np.savetxt('guide1.txt',env.global_guide_map)
np.savetxt('guidex1.txt',resultx1)
np.savetxt('guidey1.txt',resulty1)
#x2 = np.loadtxt("guidex1.txt")

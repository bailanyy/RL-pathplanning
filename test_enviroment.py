'''
Author: yy.yy 729853861@qq.com
Date: 2024-06-20 19:52:36
LastEditors: yy.yy 729853861@qq.com
LastEditTime: 2024-06-20 21:07:38
FilePath: \dqnc:\workspace\dissertation_project\test_enviroment.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import gymnasium as gym

env = gym.make("voxelgym2D:onestep-v0")
observation, info = env.reset(seed=123456)

done = False
env.render()
while not done:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    #done = terminated or truncated
    

env.close()
'''
Author: yy.yy 729853861@qq.com
Date: 2024-06-20 21:31:11
LastEditors: yy.yy 729853861@qq.com
LastEditTime: 2024-06-21 20:04:54
FilePath: \dqn\A2C.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import gym

from a2c_model import Actor_Critic
from a2c_model import Actor_Critic_DE
from a2c_model import Actor_Critic_QDE
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    model = Actor_Critic_DE(env)  #实例化Actor_Critic算法类
    reward = []
    for episode in range(200):
  
        s = env.reset()[0]  #获取环境状态
        env.render()     #界面可视化
        done = False     #记录当前回合游戏是否结束
        ep_r = 0
        cnt = 0
        while not done:
            # 通过Actor_Critic算法对当前环境做出行动
            if cnt > 200:
                done = True
            #a,log_prob,p = model.get_action(s)
            a,log_prob = model.get_action(s)
            
            # 获得在做出a行动后的最新环境
       
            s_,rew,done, _, _ = env.step(a)
            # get all posible next state
            #计算当前reward
            
            ep_r += rew

            #训练模型
            model.learn(log_prob,s,s_,rew)

            #更新环境
            s = s_
            cnt += 1
        if ep_r > 300:
            ep_r = 300
        reward.append(ep_r)
        print(f"episode:{episode} ep_r:{ep_r}")
    plt.plot(reward)
    plt.show()

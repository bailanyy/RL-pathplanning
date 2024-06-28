'''
Author: yy.yy 729853861@qq.com
Date: 2024-05-21 20:31:56
LastEditors: yy.yy 729853861@qq.com
LastEditTime: 2024-06-20 20:50:46
FilePath: \dqn\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import gym
from dqn import DQN, ReplayBuffer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

# ------------------------------- #
# 全局变量
# ------------------------------- #

capacity = 3000  # 经验池容量
lr = 0.001  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.9  # 贪心系数
target_update = 20  # 目标网络的参数的更新频率
batch_size = 32
n_hidden = 24  # 隐含层神经元个数
min_size = 2000  # 经验池超过1000后再训练
return_list = []  # 记录每个回合的回报

# 加载环境
env = gym.make("MountainCar-v0", render_mode="human")
n_states = env.observation_space.shape[0]  # 4
n_actions = env.action_space.n  # 2

# 实例化经验池
replay_buffer = ReplayBuffer(capacity)
# 实例化DQN
agent = DQN(n_states=n_states,
            n_hidden=n_hidden,
            n_actions=n_actions,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device,
        )
count = 0
# 训练模型
for i in range(400):  # 100回合
    # 每个回合开始前重置环境
    state = env.reset()[0]  # len=4
    # 记录每个回合的回报
    episode_return = 0
    done = False
    print('start')
    # 打印训练进度，一共10回合
    with tqdm(total=100, desc='Iteration %d' % i) as pbar:
        print('why')
        while True:
            
            # 获取当前状态下需要采取的动作
            if replay_buffer.size() < min_size:
                action = agent.take_action_ER(state)
            else:
                action = agent.take_action(state,count)
            # 更新环境
            next_state, reward, done, _, _ = env.step(1)
            # 添加经验池
            reward = next_state[0]+0.5
            if next_state[0] >= -0.5:
                reward += 0.5  
                if next_state[0] >= 0.5:
                    reward += next_state[0] * 10
                
            else:
                reward =0
                
            

            replay_buffer.add(state, action, reward, next_state, done)
            # 更新当前状态
            state = next_state
            # 更新回合回报
            episode_return += reward

            # 当经验池超过一定数量后，训练网络
            if replay_buffer.size() > min_size:
                # 从经验池中随机抽样作为训练集
                print('start study')
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                

                # 构造训练集
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'dones': d,
                }
                # 网络更新
                agent.update(transition_dict)
            # 找到目标就结束
            if done: break
         #   if episode_return < -199 : break
        count += 1 
        # 记录每个回合的回报
        return_list.append(episode_return)

        # 更新进度条信息
        pbar.set_postfix({
            'return': '%.3f' % return_list[-1]
        })
        pbar.update(10)

# 绘图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN Returns')
plt.show()




#测试模型稳定性
state = env.reset()[0]
for i in range(50):

    states = torch.tensor(state, dtype=torch.float)
    action_value = agent.q_net(states)
    action = action_value.argmax().item()
    next_state, reward, done, _, _ = env.step(action)
    state = next_state

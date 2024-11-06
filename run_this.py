import random
import numpy as np
from RL_DDQN import DDQN, ReplayBuffer, PrioritizedReplayBuffer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from Envs import DCMicrogrid
import csv
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

#-----------------------------define--------------------------------------
# 参数解析
def HyperParser():
    parser = argparse.ArgumentParser(description='实验参数')
    parser.add_argument('--exp_idx', default=0, help="实验索引")
    return parser.parse_args()

args = HyperParser()

# 保存CSV数据
def save_to_csv(filename, *args):
    with open(filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # 写入表头
        csv_writer.writerow([
            'Episodes', 'P1', 'P2', 'P3', 'P4', 
            'Q1', 'Q2', 'Q3', 'Q4', 
            'Cost1', 'Cost2', 'Cost3', 'Cost4',
            'loss1', 'loss2', 'loss3', 'loss4', 
            'Return1', 'Return2', 'Return3', 'Return4'])
        # 写入数据
        for i, values in enumerate(zip(*args)):
            csv_writer.writerow([i+1] + list(values))

# 定义一个通用的存储函数
def store_values(lists, values):
    for lst, val in zip(lists, values):
        lst.append(val)

# 设置随机数种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def cooperative_update(agents, replay_buffers, batch_size, min_size):
    for agent, replay_buffer in zip(agents, replay_buffers):
        if replay_buffer.size() > min_size:
            s, a, r, ns, d, weights, indices = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': s,
                'actions': a,
                'next_states': ns,
                'rewards': r,
                'dones': d,
                'weights': weights,
                'indices': indices,
            }
            agent.update(transition_dict)

            # 更新优先级
            new_priorities = agent.get_td_errors(transition_dict)
            replay_buffer.update_priorities(indices, new_priorities)
            
            # 各个智能体共享经验
            for other_agent in agents:
                if other_agent != agent:
                    other_agent.update(transition_dict)

#-----------------------------define--------------------------------------
   
# 检查是否使用GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------- #
# 全局变量
# ------------------------------- #
capacity = 5000  # 经验池容量
lr = 5e-3  # 学习率
L2_regularization_factor = 1e-4
gamma = 0.9  # 折扣因子
epsilon = 1  # 贪心系数
epsilon_rate = 0.01
target_update = 100  # 目标网络的参数的更新频率
batch_size = 64
n_hidden1 = 256  # 隐含层神经元个数
n_hidden2 = 128  # 隐含层神经元个数
min_size = 200  # 经验池超过200后再训练

# 初始化数据列表
return_list1, return_list2, return_list3, return_list4 = [], [], [], []
P1_list, P2_list, P3_list, P4_list, Pmean_list = [], [], [], [], []
Cost1_list, Cost2_list, Cost3_list, Cost4_list = [], [], [], []
loss1_list, loss2_list, loss3_list, loss4_list = [], [], [], []
Q1_list, Q2_list, Q3_list, Q4_list = [], [], [], []
Act1_list, Act2_list, Act3_list, Act4_list = [], [], [], []
average_return1_list, average_return2_list, average_return3_list, average_return4_list = [], [], [], []
V1_list, V2_list, V3_list, V4_list = [], [], [], []
cost_sum_list = []

#------------------------------------------
episodes = 500 #共训练几个episodes
iter_max = 500 #共训练几个iterations
dqn_type = 'DoubleDQN'
#------------------------------------------

# 实验保存路径
save_path = Path(f'OptimalControlExps/exp{args.exp_idx}')
save_path.mkdir(exist_ok=True, parents=True)
data_filename = save_path / f'data_exp_{episodes}_episodes.csv'

# 设置随机数种子
set_random_seed(114514)

# 加载环境
microgrid = DCMicrogrid()
n_states = 9
n1_actions, n2_actions, n3_actions, n4_actions = 3, 3, 3, 3

# 实例化经验池
'''replay_buffer1 = ReplayBuffer(capacity)
replay_buffer2 = ReplayBuffer(capacity)
replay_buffer3 = ReplayBuffer(capacity)
replay_buffer4 = ReplayBuffer(capacity)'''
replay_buffer1 = PrioritizedReplayBuffer(capacity)
replay_buffer2 = PrioritizedReplayBuffer(capacity)
replay_buffer3 = PrioritizedReplayBuffer(capacity)
replay_buffer4 = PrioritizedReplayBuffer(capacity)

# 实例化多个DQN
agent1 = DDQN(n_states=n_states,
            n_hidden1=n_hidden1,
            n_hidden2=n_hidden2,
            n_actions=n1_actions,
            learning_rate=lr,
            L2_regularization_factor=L2_regularization_factor,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_rate = epsilon_rate,
            target_update=target_update,
            device=device,
            dqn_type=dqn_type,
        )
agent2 = DDQN(n_states=n_states,
            n_hidden1=n_hidden1,
            n_hidden2=n_hidden2,
            n_actions=n2_actions,
            learning_rate=lr,
            L2_regularization_factor=L2_regularization_factor,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_rate = epsilon_rate,
            target_update=target_update,
            device=device,
            dqn_type=dqn_type,
        )
agent3 = DDQN(n_states=n_states,
            n_hidden1=n_hidden1,
            n_hidden2=n_hidden2,
            n_actions=n3_actions,
            learning_rate=lr,
            L2_regularization_factor=L2_regularization_factor,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_rate = epsilon_rate,
            target_update=target_update,
            device=device,
            dqn_type=dqn_type,
        )
agent4 = DDQN(n_states=n_states,
            n_hidden1=n_hidden1,
            n_hidden2=n_hidden2,
            n_actions=n4_actions,
            learning_rate=lr,
            L2_regularization_factor=L2_regularization_factor,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_rate = epsilon_rate,
            target_update=target_update,
            device=device,
            dqn_type=dqn_type,
        )

'''if args.resume:
    # 加载模型参数
    for idx, agent in enumerate([agent1, agent2, agent3, agent4]):
        
        agent.load_model(f'agent{idx}xxxx.pt')'''

# 训练模型
iteri = 0
for i in range(episodes):  # episodes个回合
    iter = 0
    # 每个回合开始前重置环境
    state1 = microgrid.reset(1)  # len=7
    state2 = microgrid.reset(2)
    state3 = microgrid.reset(3)
    state4 = microgrid.reset(4)

    # 记录每个回合的回报
    episode_return1 = 0
    episode_return2 = 0
    episode_return3 = 0
    episode_return4 = 0
    done = False
    
    if iteri == 0:
        cost_sum_list.append(state1[8] + state2[8] + state3[8] + state4[8])

    # 打印训练进度
    with tqdm(total=iter_max, desc=f'Episode {i+1}/{episodes}') as pbar:
        while not done and iter < iter_max:
            
            iter += 1
            iteri += 1
            
            # 更新Plist
            P1_list.append(state1[1])
            P2_list.append(state1[3])
            P3_list.append(state1[5])
            P4_list.append(state2[5])
            Pmean_list.append((state1[1] + state1[3] + state1[5] + state2[5])/4)

            #更新CostList
            Cost1_list.append(state1[8])
            Cost2_list.append(state2[8])
            Cost3_list.append(state3[8])
            Cost4_list.append(state4[8])

            V1_list.append(state1[0])
            V2_list.append(state1[2])
            V3_list.append(state1[4])
            V4_list.append(state2[4])

            # 获取当前状态下需要采取的动作
            action1 = agent1.take_action(state1, i)
            action2 = agent2.take_action(state2, i)
            action3 = agent3.take_action(state3, i)
            action4 = agent4.take_action(state4, i)
            
            actions = [action1, action2, action3, action4]

            Act1_list.append(action1-1)
            Act2_list.append(action2-1)
            Act3_list.append(action3-1)
            Act4_list.append(action4-1)

            # 更新环境
            next_states, rewards, done, _ = microgrid.step(actions)

            next_state1 = next_states[0]
            next_state2 = next_states[1]
            next_state3 = next_states[2]
            next_state4 = next_states[3]

            reward1 = rewards[0]
            reward2 = rewards[1]
            reward3 = rewards[2]
            reward4 = rewards[3]

            # 添加经验池
            replay_buffer1.add(state1, action1, reward1, next_state1, done)
            replay_buffer2.add(state2, action2, reward2, next_state2, done)
            replay_buffer3.add(state3, action3, reward3, next_state3, done)
            replay_buffer4.add(state4, action4, reward4, next_state4, done)
            # 更新当前状态
            state1 = next_state1
            state2 = next_state2
            state3 = next_state3
            state4 = next_state4
            # 更新回合回报
            episode_return1 += reward1
            episode_return2 += reward2
            episode_return3 += reward3
            episode_return4 += reward4

            # 当经验池超过一定数量后，训练网络            
            replay_buffers = [replay_buffer1, replay_buffer2, replay_buffer3, replay_buffer4]
            agents = [agent1, agent2, agent3, agent4]

            '''#自己更新
            for replay_buffer, agent in zip(replay_buffers, agents):
                if replay_buffer.size() > min_size:
                    # 从经验池中随机抽样作为训练集
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

                    # 获取代理对象
                    agent1 = agents[0]
                    agent2 = agents[1]
                    agent3 = agents[2]
                    agent4 = agents[3]'''
            
            #共享更新
            cooperative_update(agents, replay_buffers, batch_size, min_size)

            # 获取代理对象
            agent1 = agents[0]
            agent2 = agents[1]
            agent3 = agents[2]
            agent4 = agents[3]

            
            #更新Loss和Q
            loss1_list.append(agent1.get_loss())
            loss2_list.append(agent2.get_loss())
            loss3_list.append(agent3.get_loss())
            loss4_list.append(agent4.get_loss())

            Q1_list.append(agent1.get_Q_values())
            Q2_list.append(agent2.get_Q_values())
            Q3_list.append(agent3.get_Q_values())
            Q4_list.append(agent4.get_Q_values())  

            #Power
            writer.add_scalars("Power", {
                'P1': P1_list[-1], 'P2': P2_list[-1],
                'P3': P3_list[-1], 'P4': P4_list[-1]
            }, global_step=iteri)  

            #cost
            writer.add_scalars("Cost", {
                'Cost1': Cost1_list[-1], 'Cost2': Cost2_list[-1],
                'Cost3': Cost3_list[-1], 'Cost4': Cost4_list[-1],
                'Cost_sum': Cost1_list[-1] + Cost2_list[-1] + Cost3_list[-1] + Cost4_list[-1]
            }, global_step=iteri)

            #V
            writer.add_scalars("Voltage", {
                'V1': V1_list[-1], 'V2': V2_list[-1],
                'V3': V3_list[-1], 'V4': V4_list[-1],
            }, global_step=iteri)

            # 更新日志和进度条
            pbar.update(1)  # 每次迭代更新进度条
            pbar.set_postfix({'iter': iter})    
            
        # 记录每个回合的回报
        return_list1.append(episode_return1)
        return_list2.append(episode_return2)
        return_list3.append(episode_return3)
        return_list4.append(episode_return4)
        
        # 记录每个回合的最终回报
        cost_sum = state1[8] + state2[8] + state3[8] + state4[8]
        cost_sum_list.append(cost_sum)
        
        # TensorBoard记录
        writer.add_scalars("Return", {
            'R1': episode_return1, 'R2': episode_return2,
            'R3': episode_return3, 'R4': episode_return4
        }, global_step=i)

        writer.add_scalar("Cost/Total_Cost_Sum", cost_sum_list[-2], global_step=i)

        
        
        
writer.close()

#-------------------------------------------------------------------------------
# 保存数据和模型
max_len = max(len(P1_list), len(Q1_list), len(Cost1_list), len(loss1_list), len(return_list1))

# 如果 return_list 比较短，可以填充默认值
return_list1.extend([None] * (max_len - len(return_list1)))
return_list2.extend([None] * (max_len - len(return_list2)))
return_list3.extend([None] * (max_len - len(return_list3)))
return_list4.extend([None] * (max_len - len(return_list4)))
P1_list.extend([None] * (max_len - len(return_list1)))
P2_list.extend([None] * (max_len - len(return_list1)))
P3_list.extend([None] * (max_len - len(return_list1)))
P4_list.extend([None] * (max_len - len(return_list1)))
Cost1_list.extend([None] * (max_len - len(return_list1)))
Cost2_list.extend([None] * (max_len - len(return_list1)))
Cost3_list.extend([None] * (max_len - len(return_list1)))
Cost4_list.extend([None] * (max_len - len(return_list1)))

print(f"Saving data to: {data_filename}...")
save_to_csv(data_filename, P1_list, P2_list, P3_list, P4_list, Q1_list, Q2_list, Q3_list, Q4_list, 
            Cost1_list, Cost2_list, Cost3_list, Cost4_list, loss1_list, loss2_list, loss3_list, loss4_list, 
            return_list1, return_list2, return_list3, return_list4)

for idx, agent in enumerate(agents, start=1):
    agent.save_model(save_path / f"models_saved/agent{idx}_episode_{episodes}.pt", episodes)

print(f"所有数据保存到 {data_filename}")
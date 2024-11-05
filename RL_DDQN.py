import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import random
from Envs import DCMicrogrid
import os
from torch.optim.lr_scheduler import StepLR

# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer():
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)
    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    # 目前队列长度
    def size(self):
        return len(self.buffer)
   

# --------------------------------------- #
# 优先经验回放池
# --------------------------------------- #

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6):
        super().__init__(capacity)
        self.alpha = alpha
        self.capacity = capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # 优先级数组
    
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        super().add(state, action, reward, next_state, done)
        self.priorities[len(self.buffer) - 1] = max_priority
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        # 使用绝对值来防止负数优先级
        priorities = np.abs(priorities)
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        state, action, reward, next_state, done = zip(*samples)
        return np.array(state), action, reward, np.array(next_state), done, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #

class Net(nn.Module):
    # 构造只有2个隐含层的网络
    def __init__(self, n_states, n_hidden1, n_hidden2, n_actions):
        super(Net, self).__init__()
        # [b,n_states]-->[b,n_hidden]
        self.fc1 = nn.Sequential(nn.Linear(n_states, n_hidden1), nn.ReLU(True))
        # [b,n_hidden]-->[b,n_actions]
        self.fc2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2), nn.ReLU(True))
        self.fc3 = nn.Sequential(nn.Linear(n_hidden2, n_actions))
    # 前传
    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #

class DDQN():
    #（1）初始化
    def __init__(self, n_states, n_hidden1, n_hidden2, n_actions,
                 learning_rate, L2_regularization_factor, gamma, epsilon, epsilon_rate,
                 target_update, device, dqn_type):
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden1 = n_hidden1  # 隐含层神经元
        self.n_hidden2 = n_hidden2
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.L2_regularization_factor = L2_regularization_factor
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有epsilon的概率探索
        self.epsilon_rate = epsilon_rate
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        self.dqn_type = dqn_type
        # 计数器，记录迭代次数
        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
        self.q_net = Net(self.n_states, self.n_hidden1, self.n_hidden2, self.n_actions).to(device)
        # 实例化目标网络
        self.target_q_net = Net(self.n_states, self.n_hidden1, self.n_hidden2, self.n_actions).to(device)

        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate, weight_decay=L2_regularization_factor)

        # 引入学习率衰减，每隔100步衰减学习率
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.loss = None
        self.Q_values = None


    #（2）动作选择
    def take_action(self, state, epoch):
        # self.q_net.eval()
        # 维度扩充，给行增加一个维度，并转换为张量shape=[1,7]
        state = torch.Tensor(state[np.newaxis, :]).to(self.device)
        # 如果小于该值就随机探索
        # if epoch >= 10:
        #     self.epsilon = self.epsilon - epoch*self.epsilon_rate
        #epsilon = self.epsilon - epoch*self.epsilon_rate
        epsilon = self.epsilon*(0.995 ** epoch)
        if np.random.random() < max(epsilon, 0.05):  # 从epsilon衰减到0.1
            # 随机选择一个动作
            action = np.random.randint(self.n_actions)
        # 如果大于该值就取最大的值对应的索引
        else:
            # 前向传播获取该状态对应的动作的reward
            actions_value = self.q_net(state)
            # 获取reward最大值对应的动作索引
            action = actions_value.argmax().item()  # int
        # self.q_net.train()
        return action


    #（3）网络训练
    def update(self, transition_dict):  # 传入经验池中的batch个样本
        # 获取当前时刻的状态 array_shape=[b,4]
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        # 下一时刻的状态 array_shape=[b,4]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        # 输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
        weights = torch.tensor(transition_dict['weights'], dtype=torch.float).to(self.device)

        # 使用当前 Q 网络计算动作值
        q_values = self.q_net(states).gather(1, actions)  # [b,1]
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
        # 使用目标 Q 网络计算下一个状态的最大动作值
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)

        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)

        self.Q_values = q_targets[0].item() if q_targets[0] is not None else None
        
        # TD 误差计算
        td_errors = q_targets - q_values
        dqn_loss = (weights * td_errors ** 2).mean()  # 使用 weights 对 TD 误差加权

        '''# 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))'''
        #-----------------------------------
        self.loss = dqn_loss
        #-----------------------------------
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.optimizer.step()

        # 更新学习率
        self.scheduler.step()

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.count += 1

    def get_td_errors(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 获取当前状态下的 Q 值
        q_values = self.q_net(states).gather(1, actions)

        # 获取下一个状态的最大 Q 值
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        
        # 计算目标 Q 值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # TD 误差是 Q 目标值与 Q 当前值的差值
        td_errors = q_targets - q_values
        return td_errors.detach().cpu().numpy()  # 返回 numpy 数组以便更新优先级

    #-----------------------------------
    # 新增获取损失值和Q值的方法
    def get_loss(self):
        return self.loss.item() if self.loss is not None else None
        # return self.loss

    def get_Q_values(self):
        if self.Q_values is not None:
            return self.Q_values
    #-----------------------------------
    
    #-----------------------------------
     #保存和加载模型
    def save_model(self, model_path, episodes):
        # 创建文件夹
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # 保存模型参数
        checkpoint = {
            'model': self.q_net.state_dict(),
            'episodes': episodes
        }
        torch.save(checkpoint, model_path)
        print("Model saved successfully.")

    def load_model(self, model_path):
        # 加载模型参数
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.q_net.load_state_dict(checkpoint['model'])
            # self.seed = checkpoint['seed']
            print("Model loaded successfully.")
        else:
            print("Model path does not exist.")
    #-----------------------------------

if __name__ == '__main__':

    capacity = 1000  # 经验池容量
    lr = 1e-3  # 学习率
    gamma = 0.9  # 折扣因子
    epsilon = 0.9  # 贪心系数
    target_update = 200  # 目标网络的参数的更新频率
    batch_size = 32

    n_hidden1 = 128  # 隐含层神经元个数
    n_hidden2 = 256  # 隐含层神经元个数
    n_states=7
    n2_actions=3

    min_size = 200  # 经验池超过200后再训练
    return_list2 = []  # 记录每个回合的回报

    episodes = 100 #共训练几个episodes
    episode_num = 0 #第几个episode

    device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")

    microgrid = DCMicrogrid()

    agent2 = DDQN(n_states=n_states,
                n_hidden1=n_hidden1,
                n_hidden2=n_hidden2,
                n_actions=n2_actions,
                learning_rate=lr,
                gamma=gamma,
                epsilon=epsilon,
                target_update=target_update,
                device=device,
            )

    state2 = microgrid.reset(2)
    print(state2)
    print(state2.shape)
    action2=agent2.take_action(state2)
    print(action2)
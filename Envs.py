import random
from sympy import *
import numpy as np
from gym import spaces

class DCMicrogrid:
    def __init__(self):
        self.Vout1 = 800
        self.Vout2 = 800
        self.Vout3 = 800
        self.Vout4 = 800
        self.V01 = 800
        self.V02 = 800
        self.V03 = 800
        self.V04 = 800
        self.Droop1 = 0.002
        self.Droop2 = 0.002
        self.Droop3 = 0.002
        self.Droop4 = 0.002
        self.Iload1 = 20
        self.Iload2 = 50
        self.Iload3 = 10
        self.Iload4 = 50
        self.Pref1 = 0
        self.Pref2 = 0
        self.Pref3 = 0
        self.Pref4 = 0
        self.Ru = 2
        self.Rb = 2
        self.Rl = 2
        self.Rr = 2
        self.alpha = [1/100, 1/100, 1/100, 1/100]
        self.weights = [1/1000, 1/1000, 1/1000, 1/1000]
        self.Iout1 = 0
        self.Iout2 = 0
        self.Iout3 = 0
        self.Iout4 = 0
        self.max_Vout = [1000, 1000, 1000, 1000]
        self.min_Vout = [600, 600, 600, 600]
        self.max_P = [100, 100, 100, 100]
        self.min_P = [0, 0, 0, 0]
        self.low = np.array([self.min_Vout, self.min_P], dtype=np.float32)
        self.high = np.array([self.max_Vout, self.max_P], dtype=np.float32)
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.state1 = None
        self.state2 = None
        self.state3 = None
        self.state4 = None

        self.action_list1, self.action_list2, self.action_list3, self.action_list4 = [1], [1], [1], [1]
        self.cost_list1, self.cost_list2, self.cost_list3, self.cost_list4 = [0], [0], [0], [0]

        self.a1, self.b1, self.c1 = 0.5, 0.3, 0.1
        self.a2, self.b2, self.c2 = 0.6, 0.4, 0.2
        self.a3, self.b3, self.c3 = 0.7, 0.5, 0.3
        self.a4, self.b4, self.c4 = 0.8, 0.6, 0.4 

        self.Problem ='Lowest Cost'

        
    def AveragePower_Reward(self, Pself, Pnb1, Pnb2):
        Pmean = (Pself + Pnb1 + Pnb2)/3
        if abs(Pself - Pmean) == 0:
            reward = 200
        if abs(Pself - Pmean) <= 0.01:
            reward = 100
        if abs(Pself - Pmean) <= 1:
            reward = 5
        else:
            if Pself < 0:
                reward = -100
            else:
                reward = - abs(Pself - Pmean)
        return reward
    
    def LowestCost_Reward_Self_Competition(self, Vself, costself, costselfLast, costnb1, costnb1Last, costnb2, costnb2Last):
        # 计算每个智能体的成本差异
        self_diff = costselfLast - costself
        nb1_diff = costnb1Last - costnb1
        nb2_diff = costnb2Last - costnb2

        # 初始化 bias 为 0
        bias = 0

        # 自身成本变化
        if self_diff >= 0:
            bias += np.log(max(self_diff, 1))  # 自身成本降低时增加奖励

        # 邻居1成本变化
        if nb1_diff >= 0:
            bias += 0.8 * np.log(max(nb1_diff, 1))  # 邻居1成本降低时增加中等奖励

        # 邻居2成本变化
        if nb2_diff >= 0:
            bias += 0.8 * np.log(max(nb2_diff, 1))  # 邻居2成本降低时增加中等奖励

        # 自身成本增加时的惩罚
        if self_diff < 0:
            bias -= np.log(max(-self_diff, 1))  # 自身成本增加时惩罚

        if nb1_diff < 0:
            bias -= np.log(max(-nb1_diff, 1)) 

        if nb2_diff < 0:
            bias -= np.log(max(-nb2_diff, 1)) 

        reward = bias      #自我竞争奖励
        return reward
    
    def LowestCost_Reward_Neighbor_Competition(self, costself, costnb1, costnb2):
        # 计算总成本
        total_cost = costself + costnb1 + costnb2
        reward = -total_cost  # 总成本越小，奖励越高
        return reward
    
    def step(self, actions): #第几个agent
    
        self.Vout1, P1, self.Vout2, P2, self.Vout3, P3, self.Iload1, last_action1, Cost1Last = self.state1
        self.Vout1, P1, self.Vout2, P2, self.Vout4, P4, self.Iload2, last_action2, Cost2Last = self.state2
        self.Vout1, P1, self.Vout3, P3, self.Vout4, P4, self.Iload3, last_action3, Cost3Last = self.state3
        self.Vout2, P2, self.Vout3, P3, self.Vout4, P4, self.Iload4, last_action4, Cost4Last = self.state4

        action1 = actions[0]
        action2 = actions[1]
        action3 = actions[2]
        action4 = actions[3]

        self.action_list1.append(action1)
        self.action_list2.append(action2)
        self.action_list3.append(action3)
        self.action_list4.append(action4)

        last_action1 = self.action_list1[-2]
        last_action2 = self.action_list2[-2]
        last_action3 = self.action_list3[-2]
        last_action4 = self.action_list4[-2]

        self.Pref1 = self.Pref1 + 1000*(action1-1)  #更新Pref1
        self.Pref2 = self.Pref2 + 1000*(action2-1)  #更新Pref2    
        self.Pref3 = self.Pref3 + 1000*(action3-1)  #更新Pref3       
        self.Pref4 = self.Pref4 + 1000*(action4-1)  #更新Pref4

        Vref1 = (P1 - self.Pref1) * self.Droop1 + self.V01
        Vref2 = (P2 - self.Pref2) * self.Droop2 + self.V02
        Vref3 = (P3 - self.Pref3) * self.Droop3 + self.V03
        Vref4 = (P4 - self.Pref4) * self.Droop4 + self.V04
        
        self.Vout1 = Vref1        
        self.Vout2 = Vref2        
        self.Vout3 = Vref3        
        self.Vout4 = Vref4

        Iu = (self.Vout1 - self.Vout2) / self.Ru
        Il = (self.Vout1 - self.Vout3) / self.Rl
        Ir = (self.Vout2 - self.Vout4) / self.Rr
        Ib = (self.Vout3 - self.Vout4) / self.Rb
        
        self.Iout1 = self.Iload1 + Iu + Il
        self.Iout2 = self.Iload2 - Iu + Ir
        self.Iout3 = self.Iload3 + Ib - Il
        self.Iout4 = self.Iload4 - Ib - Ir

        P1 = self.Vout1 * self.Iout1 / 1000 #单位KW         
        P2 = self.Vout2 * self.Iout2 / 1000 #单位KW       
        P3 = self.Vout3 * self.Iout3 / 1000 #单位KW       
        P4 = self.Vout4 * self.Iout4 / 1000 #单位KW

        P1 = max(0, P1)
        P2 = max(0, P2)
        P3 = max(0, P3)
        P4 = max(0, P4)

        Cost1 = self.a1 * (P1 ** 2) + self.b1 * P1 + self.c1
        Cost2 = self.a2 * (P2 ** 2) + self.b2 * P2 + self.c2
        Cost3 = self.a3 * (P3 ** 2) + self.b3 * P3 + self.c3
        Cost4 = self.a4 * (P4 ** 2) + self.b4 * P4 + self.c4

        self.cost_list1.append(Cost1)
        self.cost_list2.append(Cost2)
        self.cost_list3.append(Cost3)
        self.cost_list4.append(Cost4)

        Cost1Last = self.cost_list1[-2]
        Cost2Last = self.cost_list2[-2]
        Cost3Last = self.cost_list3[-2]
        Cost4Last = self.cost_list4[-2]

        self.state1 = (self.Vout1, P1, self.Vout2, P2, self.Vout3, P3, self.Iload1, last_action1, Cost1Last)
        self.state2 = (self.Vout1, P1, self.Vout2, P2, self.Vout4, P4, self.Iload2, last_action2, Cost2Last)
        self.state3 = (self.Vout1, P1, self.Vout3, P3, self.Vout4, P4, self.Iload3, last_action3, Cost3Last)
        self.state4 = (self.Vout2, P2, self.Vout3, P3, self.Vout4, P4, self.Iload4, last_action4, Cost4Last)

        Pmean1 = (P1 + P2 + P3)/3
        Pmean2 = (P1 + P2 + P4)/3
        Pmean3 = (P1 + P3 + P4)/3
        Pmean4 = (P2 + P3 + P4)/3

        #------------------reward------------------------------------
        if self.Problem == 'Average Power':
            done = bool(
                abs(P1 - Pmean1) == 0
                and abs(P2 - Pmean2) == 0
                and abs(P3 - Pmean3) == 0
                and abs(P4 - Pmean4) == 0
            )

            # if not done:
            reward1 = self.AveragePower_Reward(P1, P2, P3)
            reward2 = self.AveragePower_Reward(P2, P1, P4)
            reward3 = self.AveragePower_Reward(P3, P1, P4)
            reward4 = self.AveragePower_Reward(P4, P2, P3)          

        elif self.Problem == 'Lowest Cost':
            done = False
            
            #if not done:
            reward1 = self.LowestCost_Reward_Self_Competition(self.Vout1, Cost1, Cost1Last, Cost2, Cost2Last, Cost3, Cost3Last)
            reward2 = self.LowestCost_Reward_Self_Competition(self.Vout2, Cost2, Cost2Last, Cost1, Cost1Last, Cost4, Cost4Last)
            reward3 = self.LowestCost_Reward_Self_Competition(self.Vout3, Cost3, Cost3Last, Cost1, Cost1Last, Cost4, Cost4Last)
            reward4 = self.LowestCost_Reward_Self_Competition(self.Vout4, Cost4, Cost4Last, Cost2, Cost2Last, Cost3, Cost3Last)

            '''reward1 = self.LowestCost_Reward_Neighbor_Competition(Cost1, Cost2, Cost3)
            reward2 = self.LowestCost_Reward_Neighbor_Competition(Cost2, Cost1, Cost4)
            reward3 = self.LowestCost_Reward_Neighbor_Competition(Cost3, Cost1, Cost4)
            reward4 = self.LowestCost_Reward_Neighbor_Competition(Cost4, Cost2, Cost3)'''
        
        #--------------------------------------------------------------

        states = [self.state1, self.state2, self.state3, self.state4]            
        rewards = [reward1, reward2, reward3, reward4]

        return np.array(states), rewards, done, {}
  

    def reset(self,num):

        # Resetting Vout values
        self.Vout1 = 800
        self.Vout2 = 800
        self.Vout3 = 800
        self.Vout4 = 800
        self.Iout1 = 30
        self.Iout2 = 30
        self.Iout3 = 30
        self.Iout4 = 30
        self.Pref1 = 0
        self.Pref2 = 0
        self.Pref3 = 0
        self.Pref4 = 0
        P1 = self.Vout1 * self.Iout1 / 1000
        P2 = self.Vout2 * self.Iout2 / 1000
        P3 = self.Vout3 * self.Iout3 / 1000
        P4 = self.Vout4 * self.Iout4 / 1000

        '''self.Iload1 = random.uniform(10, 110)
        self.Iload2 = random.uniform(10, 110)
        self.Iload3 = random.uniform(10, 110)
        self.Iload4 = random.uniform(10, 110)'''

        Cost1 = self.a1 * (P1 ** 2) + self.b1 * P1 + self.c1
        Cost2 = self.a2 * (P2 ** 2) + self.b2 * P2 + self.c2
        Cost3 = self.a3 * (P3 ** 2) + self.b3 * P3 + self.c3
        Cost4 = self.a4 * (P4 ** 2) + self.b4 * P4 + self.c4

        self.Iload1, self.Iload2, self.Iload3, self.Iload4 = 20, 50, 10, 50
        
        last_action1, last_action2, last_action3, last_action4 = 1, 1, 1, 1     
        last_cost1, last_cost2, last_cost3, last_cost4 = Cost1, Cost2, Cost3, Cost4  

        # 将生成的状态设置为环境的当前状态
        self.state1 = (self.Vout1, P1, self.Vout2, P2, self.Vout3, P3, self.Iload1, last_action1, last_cost1)
        self.state2 = (self.Vout1, P1, self.Vout2, P2, self.Vout4, P4, self.Iload2, last_action2, last_cost2)
        self.state3 = (self.Vout1, P1, self.Vout3, P3, self.Vout4, P4, self.Iload3, last_action3, last_cost3)
        self.state4 = (self.Vout2, P2, self.Vout3, P3, self.Vout4, P4, self.Iload4, last_action4, last_cost4)
        
        # 利用均匀随机分布初试化环境的状态
        self.steps_beyond_done = None
        # 设置当前步数为None
        match num:
            case 1:  
                return np.array(self.state1)
            case 2:  
                return np.array(self.state2)
            case 3:  
                return np.array(self.state3)
            case 4:  
                return np.array(self.state4)
            case _:  
                print("Other nums")
                return 0
        # 返回环境的初始化状态




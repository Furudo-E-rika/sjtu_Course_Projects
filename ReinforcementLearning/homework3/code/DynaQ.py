import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
from env import Maze
class DynaQ:
    """ Dyna-Q算法 """
    def __init__(self,
                 ncol,
                 nrow,
                 epsilon,
                 alpha,
                 gamma,
                 n_planning,
                 n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

        self.n_planning = n_planning  #执行Q-planning的次数, 对应1次Q-learning
        self.model = dict()  # 环境模型

    def check(self, state):
        if self.Q_table[state][0] == self.Q_table[state][1] == self.Q_table[state][2] == self.Q_table[state][3]:
            return True
        return False

    def take_action(self, state):  #epsilon greedy
        if np.random.random() < self.epsilon or self.check(state):
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0, a0, r, s1):
        """ ------------- Programming 1: implement the updating of the Q table for Q-learning ------------- """
        """ YOUR CODE HERE """
        self.Q_table[s0][a0] += self.alpha * (r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0][a0])
        """ ------------- Programming 1 ------------- """

    def update(self, s0, a0, r, s1):
        """ ------------- Programming 2: implement the updating of the Q table for DynaQ (you may use the function q_learning) ------------- """
        """ YOUR CODE HERE """
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = (r, s1)
        for _ in range(self.n_planning):
            s0, a0 = random.choice(list(self.model.keys()))
            r, s1 = self.model[(s0,a0)]
            self.q_learning(s0, a0, r, s1)
        """ ------------- Programming 2 ------------- """


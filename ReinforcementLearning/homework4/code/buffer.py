from collections import deque
import math
import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions import Normal

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        '''
        Args: state (ndarray (3,)), action (int), reward (float), next_state (ndarray (3,)), done (bool)

        No return
        '''
        """ ------------- Programming 5: implement the push operation of one sample ------------- """
        """ YOUR CODE HERE """
        self.buffer.append((state, action, reward, next_state, done))
        """ ------------- Programming 5 ------------- """

    def sample(self, batch_size):
        '''
        Args: batch_size (int)

        Required return: a batch of states (ndarray, (batch_size, state_dimension)), a batch of actions (list or tuple, length=batch_size),
        a batch of rewards (list or tuple, length=batch_size), a batch of next-states (ndarray, (batch_size, state_dimension)),
        a batch of done flags (list or tuple, length=batch_size)
        '''
        """ ------------- Programming 6: implement the sample operation of a batch of samples (note that to you need to satisfy 
        the format of the return as stated above to make the replay buffer compatible with other components in main.py) ------------- """
        """ YOUR CODE HERE """
        buffer_samples = random.sample(self.buffer, batch_size)
        states = np.array([s[0] for s in buffer_samples])
        actions = [s[1] for s in buffer_samples]
        rewards = [s[2] for s in buffer_samples]
        next_states = np.array([s[3] for s in buffer_samples])
        flags = [s[4] for s in buffer_samples]
        return states, actions, rewards, next_states, flags
        """ ------------- Programming 6 ------------- """

    def __len__(self):
        return len(self.buffer)
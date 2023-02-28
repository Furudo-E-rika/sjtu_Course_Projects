
import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
import pygmtools as pygm

class FeatureExtractor(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv(input_channel, 16, 5, stride=1, padding=2), nn.BatchNorm(16), nn.ReLU(), nn.Pool(2, stride=2, op='maximum'))
        self.layer2 = nn.Sequential(nn.Conv(16, 32, 5, stride=1, padding=2), nn.BatchNorm(32), nn.ReLU(), nn.Pool(2, stride=2, op='maximum'))
        

    def execute(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        feature = h2.reshape([h2.shape[0], (-1)])
        
        return feature

class MultiHeadFeatureExtractor(nn.Module):
    def __init__(self, feature_extractor, feature_size):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_size = feature_size
    
    def execute(self, X) :
        batch_size, head_size, channel_size, height, width = X.shape
        X = X.transpose(2,3,4,0,1)

        # merge batch and head
        X = X.reshape(channel_size, height, width, batch_size * head_size)
        X = X.transpose(3,0,1,2)
        output = self.feature_extractor(X)
        output = output.transpose(1,0)
        output = output.reshape(self.feature_size, batch_size, head_size)
        output = output.transpose(1,0,2)
        output = output.reshape(batch_size, self.feature_size * head_size)

        return output
        
class PermutationExtractor(nn.Module):
    def __init__(self, head_size, feature_size):
        super().__init__()
        self.head_size = head_size
        self.feature_size = feature_size
        self.fc1 = nn.Linear(feature_size * head_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, head_size**2)

    def execute(self, X):
        batch_size = X.shape[0]
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        X = self.relu(X)
        X = X.reshape((batch_size, self.head_size, self.head_size))
        X = pygm.sinkhorn(X)
        
        return X
        
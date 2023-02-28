
import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
import pygmtools as pygm
pygm.BACKEND = 'jittor'





class DeepPermNet(nn.Module):
    def __init__(self, input_channel, head_size, feature_size):
        super().__init__()

        self.head_size = head_size
        self.feature_size = feature_size
        

        self.feature_extractor = nn.Sequential(nn.Conv(input_channel, 16, 5, stride=1, padding=2),
        nn.BatchNorm(16), nn.ReLU(),
        nn.Pool(2, stride=2, op='maximum'),
        nn.Conv(16, 32, 5, stride=1, padding=2),
        nn.BatchNorm(32), nn.ReLU(), 
        nn.Pool(2, stride=2, op='maximum'),
        nn.Flatten(1)
        )

        self.permutation_extractor = nn.Sequential(nn.Linear(feature_size * head_size, 512),
        nn.ReLU(), nn.Linear(512, head_size**2)
        )
        
        

    def execute(self, X):
        batch_size, head_size, channel_size, height, width = X.shape

        X = X.reshape((-1, *X.shape[-3:]))

        feature = self.feature_extractor(X)
        
        feature = feature.reshape((batch_size, -1))
        
        permutation = self.permutation_extractor(feature)
        
        permutation = permutation.reshape((batch_size, self.head_size, self.head_size))
        permutation = pygm.linear_solvers.sinkhorn(permutation)
        
        
        return permutation
    
    
    
class Classfier(nn.Module):
    def __init__(self, input_channel, num_classes):
        super().__init__()


        self.feature_extractor = nn.Sequential(nn.Conv(input_channel, 16, 5, stride=1, padding=2),
        nn.BatchNorm(16), nn.ReLU(),
        nn.Pool(2, stride=2, op='maximum'),
        nn.Conv(16, 32, 5, stride=1, padding=2),
        nn.BatchNorm(32), nn.ReLU(), 
        nn.Pool(2, stride=2, op='maximum'),
        nn.Flatten(1)
        )
        self.fc = nn.Linear((8*8*32), num_classes)

    def execute(self, X):
        
        feature = self.feature_extractor(X)
        
        output = self.fc(feature)

        return output


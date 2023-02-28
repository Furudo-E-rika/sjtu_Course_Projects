
import jittor as jt
from jittor import init
from jittor import nn


class CNN(nn.Module):

    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv(input_channel, 16, 5, stride=1, padding=2), nn.BatchNorm(16), nn.ReLU(), nn.Pool(2, stride=2, op='maximum'))
        self.layer2 = nn.Sequential(nn.Conv(16, 32, 5, stride=1, padding=2), nn.BatchNorm(32), nn.ReLU(), nn.Pool(2, stride=2, op='maximum'))
        self.fc = nn.Linear(((8 * 8) * 32), n_classes)

    def execute(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        out = h2.reshape([h2.shape[0], (-1)])
        out = self.fc(out)
        return out
import torch.nn as nn
from torch.nn import functional as F

class CNN3(nn.Module):
	def __init__(self, in_channel):
		super().__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channel, 16, kernel_size=5, stride=1, padding=2), 
			nn.BatchNorm2d(16),
			nn.ReLU(), 
			nn.MaxPool2d(kernel_size=2, stride=2)) 

		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), 
			nn.BatchNorm2d(32), 
			nn.ReLU(), 
			nn.MaxPool2d(kernel_size=2, stride=2)) 

		self.layer3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2,stride=2))

		self.fc = nn.Linear(4*4*64, 10) 

	def forward(self, x):
		h1 = self.layer1(x)
		h2 = self.layer2(h1)
		h3 = self.layer3(h2)
		out = h3.reshape(h3.size(0), -1)
		
		out = self.fc(out)
		return out
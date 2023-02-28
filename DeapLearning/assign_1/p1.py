import jittor as jt
import matplotlib.pyplot as plt
import numpy as np
from jittor import nn, Module, init
import random

# Set seeds and define variables
np.random.seed(0)
jt.set_seed(3)
n_train = 800
n_test = 200
batch_size = 4
lr = 0.1

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_data(n):
    for i in range(n):
        x = np.random.rand(batch_size, 1)*6 - 3 # sample from (-3,3)
        y = gaussian(x,0,3)
        yield jt.float32(x), jt.float32(y)

# Three layers MLP model
class MLP(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def execute(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x


model = MLP()
optim = nn.SGD(model.parameters(), lr)

# Model training
for i,(x,y) in enumerate(get_data(n_train/batch_size)):
    pred_y = model(x)
    loss = jt.sqr(pred_y - y)
    loss_mean = loss.mean()
    optim.step(loss_mean)
    print(f"step {i}, loss = {loss_mean.numpy().sum()}")

# Model testing 
x_list = []
pred_list = []
for i in range(n_test):
    x = np.random.rand()*6 - 3
    x_list.append(x)
    x = jt.float32(x)
    pred_y = model(x)
    pred_list.append(float(pred_y))

x_and_pred = zip(x_list, pred_list)
x_and_pred = sorted(x_and_pred, key=lambda x:x[0])
x_list, pred_list = zip(*x_and_pred)

x_normal = np.arange(-3,3,0.1)
y_normal = gaussian(x_normal,0,3)

# Plotting picture
fig, ax = plt.subplots()
ax.plot(x_list, pred_list)
ax.plot(x_normal, y_normal)
ax.set(xlabel='x', ylabel='y')
ax.grid()
fig.savefig("output.png")
plt.show()
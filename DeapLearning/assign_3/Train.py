import jittor as jt
from jittor import nn, Module, init
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
from models.model import DeepPermNet
from Dataloader import Train_CIFAR, Test_CIFAR
import pygmtools as pygm
pygm.BACKEND = 'jittor'

# Loading CIFAR-10 Dataset
train_loader = Train_CIFAR(batch_size=16,shuffle=False)
test_loader = Test_CIFAR(batch_size=16,shuffle=False)


# Define model, optimizer and hyperparameters
model = DeepPermNet(input_channel=3, feature_size=512, head_size=4)
lr = 0.01
epochs = 50
## model.load_parameters(pickle.load(open("./trained_model/deeppermnet_model.pkl", "rb")))
optim = nn.SGD(model.parameters(), lr)


# Define Training and Testing 
def train(train_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0
    losses = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        #print(images.shape)
        output = model(images)
        
        #print(output.shape, labels.shape)
        loss = nn.cross_entropy_loss(output.reshape((-1, 4)), labels)
        
        optimizer.step(loss)
        losses += loss
        
        if batch_idx % 100 == 0 and batch_idx != 0: 
            print("Training in epoch {} iteration {} loss = {}".format(epoch, batch_idx, losses.numpy()[0] / 100))
            total_loss += losses
            losses = 0
    
    return total_loss.numpy()[0]

def test(test_loader, model, epoch):
    model.eval()
    total_acc = 0
    total_num = 0
    pf_acc = 0
    total_num = len(test_loader)
    for idx, (images, labels) in enumerate(test_loader): 
        batch_size = images.shape[0]
        output = model(images)
        tmp_acc = jt.sum(labels == output.argmax(dim=2)[0]) / ( batch_size * 4 )
        total_acc += tmp_acc
        pf_acc += tmp_acc
        if idx % 1000 == 0 and idx != 0:
            print('Test in epoch {} iteration {}, Acc: {}'.format(epoch, idx, pf_acc / 1000))
            pf_acc = 0
    print ('Testing result of epoch {} Acc = {}'.format(epoch, total_acc/total_num)) 
    return total_acc / total_num



if __name__ == '__main__':
    
    # Model Training    
    Best_Acc = 0
    Train_Loss_List = []
    Test_Acc_List = []
    for epoch in range(1, epochs+1):
        train_loss = train(train_loader, model, optim, epoch)
        Train_Loss_List.append(train_loss)
        Acc = test(test_loader, model, epoch)
        if Acc > Best_Acc:
            Best_Acc = Acc
        Test_Acc_List.append(Acc)
    model.save(os.path.join('./trained_model/deeppermnet_model.pkl'))

    # Train Loss and Test Accuracy Ploting
    epoch_list = np.arange(1, epochs+1)
    print("Best Acc =", Best_Acc)
    plt.figure(1)
    plt.plot(epoch_list, Train_Loss_List)
    plt.xlabel("Epochs")
    plt.ylabel("Train_loss")
    plt.savefig("./output/Perm_Train_loss.jpg")
    plt.figure(2)
    plt.plot(epoch_list, Test_Acc_List)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accurancy")
    plt.savefig("./output/Perm_Test_acc.jpg")

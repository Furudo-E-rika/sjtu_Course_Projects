import jittor as jt
from jittor import nn, Module, init
import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
from models.CNN import CNN
from models.Resnet import Resnet18
from DataLoader import Test_CIFAR, Ten_Percent_Train_CIFAR
import pickle
#if jt.has_cuda:
#    jt.flags.use_cuda = 1
#    jt.cudnn=None


    
test_loader = Test_CIFAR(batch_size=1)
masked_train_loader = Ten_Percent_Train_CIFAR(batch_size=20, data_augmentation=False)
Enhanced_train_loader = Ten_Percent_Train_CIFAR(batch_size=20, data_augmentation=True)



lr = 0.01
epochs = 50
momentum = 0.9
decay = 1e-4


def train(train_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0
    losses = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        
        output = model(images)
        loss = nn.cross_entropy_loss(output, labels)
        
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
    for idx, (images, labels) in enumerate(test_loader):
        batch_size = images.shape[0]
        output = model(images)
        pred = np.argmax(output.numpy(), axis=1)
        tmp_acc = np.sum(labels.numpy()[0] == pred) / batch_size
        total_acc += tmp_acc
        pf_acc += tmp_acc
        total_num += batch_size
        if idx % 1000 == 0 and idx != 0:
            print('Test in epoch {} iteration {}, Acc: {}'.format(epoch, idx, pf_acc / 1000))
            pf_acc = 0
    print ('Testing result of epoch {} Acc = {}'.format(epoch, total_acc/total_num)) 
    return total_acc / total_num


if __name__== '__main__':

    model_masked = CNN(input_channel=3, n_classes=10)
    optim = nn.SGD(model_masked.parameters(), lr)
    # model.load_parameters(pickle.load(open("./trained_models/model_masked.pkl", "rb")))
    Best_Acc = 0
    Train_Loss_List = []
    Test_Acc_List = []
    for epoch in range(1, epochs+1):
        train_loss = train(masked_train_loader, model_masked, optim, epoch)
        Train_Loss_List.append(train_loss)
        Acc = test(test_loader, model_masked, epoch)
        if Acc > Best_Acc:
            Best_Acc = Acc
        Test_Acc_List.append(Acc)
    model_masked.save(os.path.join('./trained_models/model_masked.pkl'))

    epoch_list = np.arange(1, epochs+1)
    print("Best Acc =", Best_Acc)
    plt.figure(1)
    plt.plot(epoch_list, Train_Loss_List)
    plt.xlabel("Epochs")
    plt.ylabel("Train_loss")
    plt.savefig("./output/masked_Train_loss.jpg")
    plt.figure(2)
    plt.plot(epoch_list, Test_Acc_List)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accurancy")
    plt.savefig("./output/masked_Test_acc.jpg")

    print("-"*20, "Applying Data Augmentation", "-"*20)

    model_enhanced = Resnet.resnet18(pretrained=True)
    
    optim = nn.SGD(model_enhanced.parameters(), lr, momentum=momentum, weight_decay=decay)
    # model.load_parameters(pickle.load(open("./trained_models/model_enhanced.pkl", "rb")))
    Best_Acc = 0
    Train_Loss_List = []
    Test_Acc_List = []
    for epoch in range(1, epochs+1):
        train_loss = train(Enhanced_train_loader, model_enhanced, optim, epoch)
        Train_Loss_List.append(train_loss)
        Acc = test(test_loader, model_enhanced, epoch)
        if Acc > Best_Acc:
            Best_Acc = Acc
        Test_Acc_List.append(Acc)
    model_enhanced.save(os.path.join('./trained_models/model_enhanced.pkl'))

    epoch_list = np.arange(1, epochs+1)
    print("Best Acc =", Best_Acc)
    plt.figure(1)
    plt.plot(epoch_list, Train_Loss_List)
    plt.xlabel("Epochs")
    plt.ylabel("Train_loss")
    plt.savefig("./output/enhanced_Train_loss.jpg")
    plt.figure(2)
    plt.plot(epoch_list, Test_Acc_List)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accurancy")
    plt.savefig("./output/enhanced_Test_acc.jpg")
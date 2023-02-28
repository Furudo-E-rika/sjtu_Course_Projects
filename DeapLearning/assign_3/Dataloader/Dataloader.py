from tkinter import image_names
from PIL import Image
import jittor as jt
from jittor.dataset import Dataset
import pickle
import numpy as np
import random


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR(Dataset):
    def __init__(self, split, data_root = '../cifar-10-batches-py', batch_size=1, shuffle=False):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        
    
        # load train data and labels
        d_list, l_list = [], []
        for file in ['/data_batch_%d' % j for j in range(1, 6)]:
            cifar_dic = unpickle(self.data_root+file)
            for i in range(len(cifar_dic[b"labels"])):
                image = np.reshape(cifar_dic[b"data"][i], (3, 32 ,32))
                image = np.transpose(image , (1, 2, 0))
                image = image.astype(float)
                d_list.append(image)
            l_list += cifar_dic[b"labels"]

        d_list = np.array (d_list, dtype='float')
        l_list = np.array (l_list, dtype='int')
        self.train_data, self.train_label = d_list, l_list
        
        # load test data and labels
        d_list, l_list = [], []
        file = '/test_batch'
        cifar_dic = unpickle(self.data_root+file)
        for i in range(len(cifar_dic[b"labels"])):
            image = np.reshape(cifar_dic[b"data"][i], (3, 32, 32))
            image = np.transpose(image, (1, 2, 0))
            image = image.astype(float)
            d_list.append(image)
        l_list += cifar_dic[b"labels"]
        
        d_list = np.array (d_list, dtype='float')
        l_list = np.array (l_list, dtype='int')
        self.test_data, self.test_label = d_list, l_list
        self.total_len = len(d_list)

        

    
    def __getitem__(self, i):

        return NotImplementedError
    

class Train_CIFAR(CIFAR):
    def __init__(self, split='train', data_root='./cifar-10-batches-py' , batch_size=1, shuffle=False):
        super(Train_CIFAR, self).__init__(split, data_root , batch_size, shuffle)
        self.total_len  = len(self.train_data)
        self.set_attrs(batch_size = self.batch_size, shuffle = self.shuffle, total_len = self.total_len)
    
    def __getitem__(self, idx):
        image, label = self.train_data[idx], self.train_label[idx]
        H, W, _ = image.shape
        middle_H, middle_W = H//2, W//2

        image_list = []
        position_list = [0,1,2,3]

        image_11 = image[0:middle_H, 0:middle_W, :]
        image_12 = image[0:middle_H, middle_W:W, :]
        image_21 = image[middle_H:H, 0:middle_W, :]
        image_22 = image[middle_H:H, middle_W:W, :]

        image_list.append(np.array(image_11).astype(np.float32).transpose(2, 0, 1))
        image_list.append(np.array(image_12).astype(np.float32).transpose(2, 0, 1))
        image_list.append(np.array(image_21).astype(np.float32).transpose(2, 0, 1))
        image_list.append(np.array(image_22).astype(np.float32).transpose(2, 0, 1))

        state = np.random.get_state()
        np.random.shuffle(image_list)
        
        np.random.set_state(state)
        np.random.shuffle(position_list)

        image_list = jt.array(np.array(image_list).astype(np.float32))
        position_list = jt.array(np.array(position_list).astype(np.int32))
        
        return image_list, position_list
        
class Test_CIFAR(CIFAR):
    def __init__(self, split='test', data_root='./cifar-10-batches-py', batch_size=1, shuffle=False):
        super(Test_CIFAR, self).__init__(split, data_root, batch_size, shuffle)
        self.total_len = len(self.test_data)
        self.set_attrs(batch_size = self.batch_size, shuffle = self.shuffle, total_len = self.total_len)

    def __getitem__(self, idx):
        image, label = self.test_data[idx], self.test_label[idx]
        H, W, _ = image.shape
        middle_H, middle_W = H//2, W//2

        image_list = []
        position_list = [0,1,2,3]

        image_11 = image[0:middle_H, 0:middle_W, :]
        image_12 = image[0:middle_H, middle_W:W, :]
        image_21 = image[middle_H:H, 0:middle_W, :]
        image_22 = image[middle_H:H, middle_W:W, :]

        image_list.append(np.array(image_11).astype(np.float32).transpose(2, 0, 1))
        image_list.append(np.array(image_12).astype(np.float32).transpose(2, 0, 1))
        image_list.append(np.array(image_21).astype(np.float32).transpose(2, 0, 1))
        image_list.append(np.array(image_22).astype(np.float32).transpose(2, 0, 1))

        state = np.random.get_state()
        np.random.shuffle(image_list)
        
        np.random.set_state(state)
        np.random.shuffle(position_list)
        
        
        image_list = jt.array(np.array(image_list).astype(np.float32))
        position_list = jt.array(np.array(position_list).astype(np.int32))
        
        return image_list, position_list
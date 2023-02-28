

from tkinter import image_names
from PIL import Image
import jittor as jt
from jittor.dataset import Dataset
import pickle
import numpy as np
from Dataloader.utils import random_bright, random_crop, random_gray, random_flip ,random_contrast, random_saturation, random_swap
import random


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR(Dataset):
    def __init__(self, split, data_root = './cifar-10-batches-py', batch_size=1, shuffle=False):
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

        self.set_attrs(batch_size = self.batch_size, shuffle = self.shuffle)

    
    def __getitem__(self, i):

        return NotImplementedError
    

class Train_CIFAR(CIFAR):
    def __init__(self, split='train', data_root='./cifar-10-batches-py' , batch_size=1, shuffle=False):
        super(Train_CIFAR, self).__init__(split, data_root , batch_size, shuffle)
        self.total_len  = len(self.train_data)
        self.set_attrs(batch_size = self.batch_size, shuffle = self.shuffle, total_len = self.total_len)
    
    def __getitem__(self, idx):
        image, label = self.train_data[idx], self.train_label[idx]
        image = np.array(image).astype(np.float).transpose(2, 0, 1)
        image = jt.array(image)
        label = jt.array(np.array(label).astype(np.int))
        return image, label
        
class Test_CIFAR(CIFAR):
    def __init__(self, split='test', data_root='./cifar-10-batches-py', batch_size=1, shuffle=False):
        super(Test_CIFAR, self).__init__(split, data_root, batch_size, shuffle)
        self.total_len = len(self.test_data)
        self.set_attrs(batch_size = self.batch_size, shuffle = self.shuffle, total_len = self.total_len)

    def __getitem__(self, idx):
        image, label = self.test_data[idx], self.test_label[idx]
        image = np.array(image).astype(np.float).transpose(2, 0, 1)
        image = jt.array(image)
        label = jt.array(np.array(label).astype(np.int))
        return image, label


class Ten_Percent_Train_CIFAR(CIFAR):
    def __init__(self, split='train', data_root='./cifar-10-batches-py', batch_size=1, shuffle=False, data_augmentation=False):
        super().__init__(split, data_root, batch_size, shuffle)
        self.data_augmentation = data_augmentation
        self.num_augmentation = 1

        ten_percent = np.argwhere(self.train_label < 5).reshape((-1))
        
        full = np.argwhere(self.train_label >= 5).reshape((-1))
        slice = np.random.choice(ten_percent, int(0.1*len(ten_percent)), replace=False)
        
        self.percent_data, self.percent_label = [], []
        for i in full :
            self.percent_data.append(self.train_data[i])
            self.percent_label.append(self.train_label[i])
        for i in slice :
            self.percent_data.append(self.train_data[i])
            self.percent_label.append(self.train_label[i])
            


        if self.data_augmentation:

            print(self.num_augmentation)
            data_copy = self.percent_data
            label_copy = self.percent_label
            Enhancement = [random_bright, random_crop, random_flip, random_gray, random_saturation, random_swap, random_contrast]
            random.shuffle(Enhancement)
            for _ in range(self.num_augmentation):
                for idx in range(len(data_copy)):
                    image = data_copy[idx]
                    label = label_copy[idx]
                    if (label < 5 and random.random() < 0.25):
                        enhance = random.choice(Enhancement)
                        image = enhance(image)
                        if (image != data_copy[idx]).any():
                            self.percent_data.append(image)
                            self.percent_label.append(label)
            

        state = np.random.get_state()
        np.random.shuffle(self.percent_data)
        
        np.random.set_state(state)
        np.random.shuffle(self.percent_label)
        
        print(len(self.percent_data))
            
        
        self.percent_data = np.array(self.percent_data, dtype = 'float')
        self.percent_label = np.array(self.percent_label, dtype = 'int')

        self.total_len = len(self.percent_label)
        

    def __getitem__(self, idx):
        image, label = self.percent_data[idx], self.percent_label[idx]

        
        image = np.array(image).astype(np.float).transpose(2, 0, 1)
        image = jt.array(image)
        label = jt.array(np.array(label).astype(np.int))
        return image, label



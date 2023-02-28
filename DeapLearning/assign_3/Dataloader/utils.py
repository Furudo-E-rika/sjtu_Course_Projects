import imp
import jittor as jt
import os
import random
import numpy as np
from jittor import transform
import json
import cv2
import math


## Data Augmentation

def random_bright(image, delta=32):
    
    delta = random.randint(-delta, delta)
    image = image + delta
    image = image.clip(min=0, max=255)
    #print('bright')
    return image

def random_swap(image):
    
    order = np.random.permutation(3)
    image = image[:,:,order]
    #print('swap')
    return image

def random_contrast(image, lower=0.5, upper=1.5):
    
    alpha = random.uniform(lower, upper)
    image = image * alpha
    image = image.clip(min=0, max=255)
    #print('contrast')
    return image

def random_saturation(image, lower=0.5, upper=1.5):
    
    image[:,:,1] = image[:,:,1] * random.uniform(lower, upper)
    image = image.clip(min=0, max=255)
    #print('saturation')
    return image

def random_gray(image):
    Gray = (image[:,:,0]+image[:,:,1]+image[:,:,2]) / 3
    image[:,:,0] = image[:,:,1] = image[:,:,2] = Gray
    #print('gray')
    return image

def random_flip(image):
    
    flipcode = random.randint(-1, 1)
    image = cv2.flip(image, flipcode)
    #print('flip')
    return image

def random_crop(image, min_scale=0.7, max_scale=1.0):
    
    scale = random.uniform(min_scale,max_scale)
    H,W,C = image.shape
    new_H = int(H * scale)
    new_W = int(W * scale)
    upper = (H-new_H) // 2
    bottom = upper + new_H
    left = (W-new_W) // 2
    right = left + new_W
    image = image[upper:bottom,left:right,:]
    image = np.array(image)
    image = cv2.resize(image,dsize=(H,W))
        
    #print('crop')

        
    return image

        
    




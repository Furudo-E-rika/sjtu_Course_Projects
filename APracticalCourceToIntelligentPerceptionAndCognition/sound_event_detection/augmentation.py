import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import random


def time_shift(feat, label, max_shift=50):
    shift_time = np.random.randint(0, max_shift)
    return np.roll(feat, shift_time), label


def add_uniform_noise(feat, label, noise_factor=0.1):
    noise = np.random.randn(*feat.shape) * noise_factor

    return feat + noise, label


def add_gaussian_noise(feat, label, mu=0, std=0.1):
    noise = np.random.normal(loc=mu, scale=std, size=feat.shape)

    return feat + noise, label


def mix_audio(feat1, target1, feat2, target2, fusion_ratio=0.5, label_operation='add'):
    if feat1.shape == feat2.shape:
        if label_operation == 'add':
            mixed_feat = feat1 * fusion_ratio + feat2 * (1 - fusion_ratio)
            mixed_label = target1 * fusion_ratio + target2 * (1 - fusion_ratio)
        elif label_operation == 'or':
            mixed_feat = feat1 * fusion_ratio + feat2 * (1 - fusion_ratio)
            mixed_label = np.logical_or(target1, target2).astype(int)
        elif label_operation == 'and':
            mixed_feat = feat1 * fusion_ratio + feat2 * (1 - fusion_ratio)
            mixed_label = np.logical_and(target1, target2).astype(int)
        else:
            raise ValueError('label_operation must be one of add, or, and')
        return mixed_feat, mixed_label
    else:
        return feat1, target1


def time_mask(feat, label, mask_rate=0.1):
    length = feat.shape[0]
    mask_len = int(mask_rate * length)
    start_time = np.random.randint(low=0, high=length-mask_len-1)
    feat[start_time:start_time+mask_len, :] = 0
    return feat, label


from typing import List, Union
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from h5py import File
from torch.utils.data import Dataset
from augmentation import *
import random

def load_dict_from_csv(file, cols, sep="\t"):
    if isinstance(file, str):
        df = pd.read_csv(file, sep=sep)
    elif isinstance(file, pd.DataFrame):
        df = file
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output


class InferenceDataset(Dataset):

    def __init__(self,
                 audio_file):
        super(InferenceDataset, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aids = list(self.aid_to_h5.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        aid = self.aids[index]
        h5_file = self.aid_to_h5[aid]
        if h5_file not in self.cache:
            self.cache[h5_file] = File(h5_file, 'r', libver='latest')
        feat = self.cache[h5_file][aid][()]
        feat = torch.as_tensor(feat).float()
        return aid, feat


class TrainDataset(Dataset):

    def __init__(self,
                 audio_file,
                 label_file,
                 label_to_idx,
                 augmentation=None,
                 augment_type='random',
                 augment_ratio=1.0,
                 audio_mix_ratio=0.8,
                 ):
        super(TrainDataset, self).__init__()
        self.aid_to_h5 = load_dict_from_csv(audio_file, ("audio_id", "hdf5_path"))
        self.cache = {}
        self.aid_to_label = load_dict_from_csv(label_file,
            ("filename", "event_labels"))
        self.aids = list(self.aid_to_label.keys())
        first_aid = self.aids[0]
        with File(self.aid_to_h5[first_aid], 'r') as store:
            self.datadim = store[first_aid].shape[-1]  # datadim = 64
        self.label_to_idx = label_to_idx
        self.augmentation = augmentation
        self.aid_list = []
        self.feat_list = []
        self.label_list = []
        self.create_dataset_list(augment_type, augment_ratio, audio_mix_ratio)

    def __len__(self):
        if self.augmentation is not None and len(self.augmentation) > 0:
            return 2 * len(self.aids)
        return len(self.aids)

    def __getitem__(self, index):
        return self.aid_list[index], self.feat_list[index], self.label_list[index]

    def create_dataset_list(self, augment_type, augment_ratio, audio_mix_ratio):

        for idx in range(len(self.aids)):
            aid = self.aids[idx]
            h5_file = self.aid_to_h5[aid]
            if h5_file not in self.cache:
                self.cache[h5_file] = File(h5_file, 'r', libver='latest')
            feat = self.cache[h5_file][aid][()]

            feat = torch.as_tensor(feat).float()
            label = self.aid_to_label[aid]
            target = torch.zeros(len(self.label_to_idx))
            for l in label.split(","):
                target[self.label_to_idx[l]] = 1

            self.aid_list.append(aid)
            self.feat_list.append(feat)
            self.label_list.append(target)

        if self.augmentation is not None and len(self.augmentation) > 0:
            for idx in range(len(self.aids)):
                aid = self.aids[idx]
                h5_file = self.aid_to_h5[aid]
                if h5_file not in self.cache:
                    self.cache[h5_file] = File(h5_file, 'r', libver='latest')
                feat = self.cache[h5_file][aid][()]
                label = self.aid_to_label[aid]
                target = np.zeros(len(self.label_to_idx))

                for l in label.split(","):
                    target[self.label_to_idx[l]] = 1

                feat, target = self.apply_augmentation(feat, target, self.augmentation, augment_type, audio_mix_ratio)
                feat = torch.as_tensor(feat).float()
                target = torch.as_tensor(target)

                self.aid_list.append(aid)
                self.feat_list.append(feat)
                self.label_list.append(target)

    def apply_single_augmentation(self, feat, target, aug, audio_mix_ratio):
        if aug == "time_shift":
            feat, target = time_shift(feat, target)
        elif aug == "time_mask":
            feat, target = time_mask(feat, target)
        elif aug == "add_uniform_noise":
            feat, target = add_uniform_noise(feat, target)
        elif aug == "add_gaussian_noise":
            feat, target = add_gaussian_noise(feat, target)
        elif aug == "mix_audio":
            # Select a random example for mixing
            idx2 = np.random.randint(len(self.aids))
            aid2 = self.aids[idx2]
            h5_file2 = self.aid_to_h5[aid2]
            if h5_file2 not in self.cache:
                self.cache[h5_file2] = File(h5_file2, 'r', libver='latest')
            feat2 = self.cache[h5_file2][aid2][()]
            label2 = self.aid_to_label[aid2]
            target2 = np.zeros(len(self.label_to_idx))
            for l in label2.split(","):
                target2[self.label_to_idx[l]] = 1
            # Mix the current example with the selected one
            feat, target = mix_audio(feat, target, feat2, target2, audio_mix_ratio)
        return feat, target

    def apply_augmentation(self, feat, target, augmentation, augment_type="random", audio_mix_ratio=0.8):
        if augment_type == "random":
            aug = random.choice(augmentation)
            feat, target = self.apply_single_augmentation(feat, target, aug, audio_mix_ratio)
        elif augment_type == "all":
            for aug in augmentation:
                feat, target = self.apply_single_augmentation(feat, target, aug, audio_mix_ratio)
        else:
            raise ValueError("Augmentation type not supported")

        return feat, target


def pad(tensorlist, batch_first=True, padding_value=0.):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    if isinstance(tensorlist[0], np.ndarray):
        tensorlist = [torch.as_tensor(arr) for arr in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist,
                                                 batch_first=batch_first,
                                                 padding_value=padding_value)
    length = [tensor.shape[0] for tensor in tensorlist]
    return padded_seq, length


def sequential_collate(return_length=True, length_idxs: List=[]):
    def wrapper(batches):
        seqs = []
        lens = []
        for idx, data_seq in enumerate(zip(*batches)):
            if isinstance(data_seq[0],
                          (torch.Tensor, np.ndarray)):  # is tensor, then pad
                data_seq, data_len = pad(data_seq)
                if idx in length_idxs:
                    lens.append(data_len)
            else:
                data_seq = np.array(data_seq)
            seqs.append(data_seq)
        if return_length:
            seqs.extend(lens)
        return seqs
    return wrapper


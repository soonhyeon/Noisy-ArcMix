import numpy as np 
import torch
from torch.utils.data import random_split


def get_accuracy(logits, y):
    pred_label = torch.argmax(logits, dim=-1)
    return torch.sum(pred_label == y)/len(pred_label)
  
def dataset_split(dataset, split_ratio):
    data_len = len(dataset)
    
    train_len = int(data_len * split_ratio)
    valid_len = data_len - train_len
    
    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

    return train_dataset, valid_dataset

# Mixup is referred to https://github.com/facebookresearch/mixup-cifar10

def mixup_data(x_wavs, x_mels, y, device, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_mels.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x_wavs = lam * x_wavs + (1 - lam) * x_wavs[index, :]
    mixed_x_mels = lam * x_mels + (1 - lam) * x_mels[index, :]
    y_a, y_b = y, y[index]
    return mixed_x_wavs, mixed_x_mels, y_a, y_b, lam

def noisy_arcmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def arcmix_criterion(criterion, pred, pred2, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred2, y_b)
import os
import glob

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
# import torchvision.transforms.functional as FT

class SRDataset(Dataset):
    def __init__(self, root='', is_train=True, downsampling_method=transforms.InterpolationMode.BILINEAR):
        self.data_root = root
        prefix = 'train' if is_train else 'eval'
        self.data_path = os.path.join(root, prefix, '*')
        self.downsampling_method = downsampling_method

        self.images = sorted(glob.glob(self.data_path))
        self.crop_size = [64, 64]

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(self.crop_size),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)])
        else:
            self.transform = transforms.Compose([]) 

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        file_name = self.images[idx]

        data = read_image(file_name).float()
        data = (data - data.min()) / (data.max() - data.min()) # normalize
        data = self.transform(data)
        
        _, h, w = data.shape
        lr_size = [h // 2, w // 2]
        lr_image = transforms.Resize(lr_size, self.downsampling_method)(data)
        return lr_image, data
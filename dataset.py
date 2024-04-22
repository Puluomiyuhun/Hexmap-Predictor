import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd

class CatDogDataset(Dataset):
 
    def __init__(self,csv_file,transform=None):
        self.data = pd.read_csv(csv_file,header=None)
        self.transform = transform
 
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx,0]
        img_name = "./training_data/" + img_name
        image = cv2.cvtColor(cv2.imread(img_name),cv2.COLOR_BGR2RGB)
        target = self.data.iloc[idx,1]
        sample = (image, target)
        if self.transform:
            sample = self.transform(sample)
 
        return sample
    
class PredictDataset(Dataset):
 
    def __init__(self,img_file,transform=None):
        self.img = cv2.imread(img_file)
        height, width, _ = self.img.shape
        self.len = int(height * width /48 / 48)
        self.transform = transform
        self.split_img = []

        for i in range(int(height / 48)):
            for j in range(int(width / 48)):
                self.split_img.append(self.img[i*48:(i+1)*48, j*48:(j+1)*48])
 
 
    def __len__(self):
        return self.len
 
    def __getitem__(self, idx):
        image = cv2.cvtColor(self.split_img[idx],cv2.COLOR_BGR2RGB)
        sample = (image, 0)
        if self.transform:
            sample = self.transform(sample)
 
        return sample
 
class Rescale(object):
 
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
 
    def __call__(self, sample):
        image,target = sample
        h,w = image.shape[:2]
        if isinstance(self.output_size,int):
            if h>w:
                new_h, new_w = self.output_size * h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image,(new_h,new_w))
        return (img,target)
 
class RandomCrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        image, target = sample
        h,w = image.shape[:2]
        new_h, new_w = self.output_size
 
        top = np.random.randint(0,h-new_h)
        left = np.random.randint(0,w-new_w)
 
        image = image[top: top+new_h, left: left+new_w]
        return (image,target)
 
class ToTensor(object):
    def __call__(self, sample):
        image, target = sample
        image = image.transpose((2,0,1))
        return (torch.from_numpy(image), int(target))
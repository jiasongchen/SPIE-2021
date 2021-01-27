#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Guc, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""
import torch
import numpy as np
from scipy.io import loadmat

class DiskSet(torch.utils.data.Dataset):
    def __init__(self, path, file, return_bon=False, return_idx=False):
        self.path=path
        self.return_bon=return_bon
        self.return_idx=return_idx
        self.filelist=[]
        with open(path+file,'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file = self.path+self.filelist[idx].rstrip()
        data = loadmat(file)
        x=data['img']
        #print(image.shape)
        x = x.reshape(1, x.shape[0], x.shape[1])#CxHxW
        
        if self.return_bon == False:
            point1 = data['disk_left']
            point2=data['disk_bot']
            point3=data['disk_right']
            point4=data['disk_up']
            s=np.concatenate((point1,point2,point3,point4),axis=0)
            #s=[point1,point2,point3,point4]
        else:
            disk_left = data['disk_left']
            disk_bot=data['disk_bot']
            disk_right=data['disk_right'] 
            disk_up=data['disk_up']
            up_bon_left=data['up_bon_left']
            up_bon_right=data['up_bon_right']
            up_bon_top=data['up_bon_top']
            bot_bon_left=data['bot_bon_left']
            bot_bon_right=data['bot_bon_right']
            bot_bon_low=data['bot_bon_low']
            
            s=np.concatenate((disk_left,disk_bot,disk_right,disk_up, 
                              up_bon_left,up_bon_right,up_bon_top,
                              bot_bon_left,bot_bon_right,bot_bon_low),axis=0)        
        
        #print(data['Patient'])
        #print(data['ElementNumber'])
        #print(point.shape)        
        #normalize image into the range of 0 to 1
        x = (x - x.min())/(x.max()-x.min())
        x = torch.tensor(x, dtype=torch.float32)
        s = torch.tensor(s, dtype=torch.float32)
        if self. return_idx== False:
            return x , s
        else:
            return x, s, idx
class DiskSet_example(torch.utils.data.Dataset):
    def __init__(self, path, file, return_idx=False):
        self.path=path
        self.return_idx=return_idx
        self.filelist=[]
        with open(path+file,'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file = self.path+self.filelist[idx].rstrip()
        data = loadmat(file)
        x=data['img']
        x = x.reshape(1, x.shape[0], x.shape[1])#CxHxW
        
        s= data['shape']
      
        x = (x - x.min())/(x.max()-x.min())
        x = torch.tensor(x, dtype=torch.float32)
        s = torch.tensor(s, dtype=torch.float32)
        if self. return_idx== False:
            return x , s
        else:
            return x, s, idx           

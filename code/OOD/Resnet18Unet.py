# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Guc, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
import matplotlib.pyplot as plt
#%%
class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(out_channels//4, out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)
        self.gn2 = nn.GroupNorm(out_channels//4, out_channels)
        self.process_x=None          
        if stride > 1 and out_channels != in_channels:
            self.process_x = nn.Sequential(nn.AvgPool2d(stride, stride),
                                           nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))
        elif stride > 1 and out_channels == in_channels:
            self.process_x = nn.AvgPool2d(stride, stride)
        elif stride == 1 and out_channels != in_channels:
            self.process_x = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        #print('out', out.shape, 'x', x.shape)
        if self.process_x is not None:
            x = self.process_x(x)
        #print('out', out.shape, 'x', x.shape)
        out = out+x
        out = self.gn2(out)        
        out = self.relu2(out)
        return out

#%% for shape regression and segmentation
class Resnet18Unet(nn.Module):
    def __init__(self, n_point, n_seg, flag=0, offset=64, output='reg_seg_rec'):
        super().__init__()
        self.n_point=n_point
        self.n_seg=n_seg
        self.flag=flag
        self.offset=offset
        self.output=output       
        #--------------------------------------------------------------
        self.e0 = nn.Sequential(nn.Conv2d(1, 32, 7, 2, 3),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(32, 64, 5, 2, 2, bias=False),
                                nn.GroupNorm(64, 64),
                                nn.LeakyReLU(inplace=True))
        self.e1 = nn.Sequential(Block(64, 128, 2),   Block(128, 128, 1))
        self.e2 = nn.Sequential(Block(128, 256, 2),  Block(256, 256, 1))
        self.e3 = nn.Sequential(Block(256, 512, 2),  Block(512, 512, 1))
        self.e4 = nn.Sequential(Block(512, 1024, 2), Block(1024, 1024, 1),
                                nn.Conv2d(1024, 1024, 2, 1, 0, bias=False))
        self.e5 = nn.Sequential(nn.Flatten(),
                                nn.GroupNorm(1, 1024),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(1024, 1024, bias=False),
                                nn.GroupNorm(1, 1024),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(1024, n_point))
        #--------------------------------------------------------------          
        if flag == 0:
            self.g4 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1, bias=False),
                                    nn.GroupNorm(1, 512),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ConvTranspose2d(512, 512, 3, 2, 1, 1, bias=False),
                                    nn.GroupNorm(2, 512),
                                    nn.LeakyReLU(inplace=True))
        else:
            self.g4 = nn.Sequential(nn.Conv2d(n_point, 1024, 1, 1, 0),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(1024, 1024, 1, 1, 0),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1, bias=False),
                                    nn.GroupNorm(1, 512),
                                    nn.LeakyReLU(inplace=True),
                                    nn.ConvTranspose2d(512, 512, 3, 2, 1, 1, bias=False),
                                    nn.GroupNorm(2, 512),
                                    nn.LeakyReLU(inplace=True))
        #--------------
        self.g3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1, bias=False),
                                nn.GroupNorm(4, 256),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(256, 256, 3, 2, 1, 1, bias=False),
                                nn.GroupNorm(4, 256),
                                nn.LeakyReLU(inplace=True))
        
        self.g2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1, bias=False),
                                nn.GroupNorm(8, 128),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(128, 128, 3, 2, 1, 1, bias=False),
                                nn.GroupNorm(8, 128),
                                nn.LeakyReLU(inplace=True))
        
        self.g1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1, bias=False),
                                nn.GroupNorm(16, 64),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(64, 64, 3, 2, 1, 1, bias=False),
                                nn.GroupNorm(16, 64),
                                nn.LeakyReLU(inplace=True))

        self.g0 = nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1, bias=False),
                                nn.GroupNorm(32, 32),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(32, 32, 7, 2, 3, 1, bias=False),
                                nn.GroupNorm(32, 32),
                                nn.LeakyReLU(inplace=True),
                                nn.ConvTranspose2d(32, n_seg, 5, 2, 2, 1))
         
    def forward(self, x):
        #print('x', x.shape)
        x0e = self.e0(x)
        #print('x0e.shape', x0e.shape)
        x1e = self.e1(x0e)
        #print('x1e.shape', x1e.shape)
        x2e = self.e2(x1e)
        #print('x2e.shape', x2e.shape)
        x3e = self.e3(x2e)
        #print('x3e.shape', x3e.shape)
        x4e = self.e4(x3e)
        #print('x4e.shape', x4e.shape)
        x5e = self.e5(x4e)
        #print('x5e.shape', x5e.shape)

        if self.flag == 0:
            x4g=x4e
        else:
            x4g=x5e.view(x5e.shape[0],x5e.shape[1],1,1)
        
        x4g=self.g4(x4g)
        #print('0', x4g.shape)

        x3g=torch.cat([x3e, x4g], dim=1)
        #print('1', x3g.shape)
        x3g=self.g3(x3g)
        #print('2', x3g.shape)

        x2g=torch.cat([x2e, x3g], dim=1)
        #print('3', x2g.shape)
        x2g=self.g2(x2g)
        #print('4', x2g.shape)

        x1g=torch.cat([x1e, x2g], dim=1)
        #print('5', x1g.shape)
        x1g=self.g1(x1g)
        #print('6', x1g.shape)
        
        x0g=torch.cat([x0e, x1g], dim=1)
        #print('7', x0g.shape)
        x0g=self.g0(x0g)
        #print('8', x0g.shape)
    
        if self.output == 'reg_seg_rec':        
            x_rec=torch.sigmoid(x0g[:,1:2])
            return x5e+self.offset, x0g[:,0:1], x_rec
        elif self.output == 'reg_seg': 
            return x5e+self.offset, x0g[:,0:1]
        elif self.output == 'reg':
            return x5e+self.offset
        elif self.output == 'seg':  
            return x0g[:,0:1]

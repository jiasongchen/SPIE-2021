# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Guc, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""

import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch.nn.functional as nnF
from Lumbar_Dataset import DiskSet
from Lumbar_Dataset import DiskSet_example
from Resnet18Unet import Resnet18Unet
from PGD import pgd_attack
from Disk_evaluate_reg import dice, poly_disk
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(0)
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(0)
#%%
def attack_loss(Mp, M):
    Mp=torch.sigmoid(Mp)
    loss_dice=1-dice(Mp, M, 'mean')
    return loss_dice
#%%
def test_adv(model, device, dataloader, noise_norm, norm_type, max_iter, step):
    model.eval()#set model to evaluation mode
    #---------------------
    dice_clean=[]
    dice_noisy=[]
    Xn_all=[]
    for batch_idx, batch_data in enumerate(dataloader):        
        X, S = batch_data[0].to(device), batch_data[1].to(device)
        M=poly_disk(S)
        #------------------
        Mp=model(X)
        Mp=(Mp>0).to(torch.float32)
        #------------------
        Xn = pgd_attack(model, X, M, noise_norm, norm_type, max_iter, step, use_optimizer=True, loss_fn=attack_loss)
        Mpn = model(Xn)
        Mpn=(Mpn>0).to(torch.float32)
        #------------------        
        dice_clean.append(dice(Mp, M).detach().cpu().numpy())
        dice_noisy.append(dice(Mpn, M).detach().cpu().numpy())    
        Xn_all.append(Xn.detach().cpu().numpy())
    #------------------
    dice_clean=np.concatenate(dice_clean)
    dice_noisy=np.concatenate(dice_noisy)
    Xn_all=np.concatenate(Xn_all)
    result={}    
    result['noise_norm']=noise_norm
    result['norm_type']=norm_type
    result['max_iter']=max_iter
    result['step']=step
    result['dice_clean']=dice_clean
    result['dice_noisy']=dice_noisy
    result['Xn']=Xn_all
    #------------------
    print('testing seg robustness, noise_norm:', noise_norm, 'norm_type', norm_type, 'max_iter', max_iter, 'step', step)
    print('dice_clean ', result['dice_clean'].mean(), '(', result['dice_clean'].std(), ')', sep='')
    print('dice_noisy ', result['dice_noisy'].mean(), '(', result['dice_noisy'].std(), ')', sep='')
    return result
#%%
def evaluate(model, device, filename, loader, norm_type):    
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device);
    if norm_type == np.inf:
        noise_norm_list=[0.01, 0.03, 0.05, 0.07, 0.1, 0.2]
    max_iter=100
    result=[]
    for noise_norm in noise_norm_list:        
        step=noise_norm/5
        result.append(test_adv(model, device, loader, noise_norm, norm_type, max_iter, step))
    filename_save=filename+'_evaluate_seg_L'+str(norm_type)
    torch.save({'result':result, 'noise_norm_list':noise_norm_list},
               filename_save+'.pt')
    print('saved', filename_save)
#%%
import argparse
if __name__ == '__main__':
    #%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--net_name', default='Resnet18Unet', type=str)
    parser.add_argument('--norm_type', default=np.inf, type=float)
    parser.add_argument('--cuda_id', default=0, type=int)
    arg = parser.parse_args()
    #%%
    path='../data/'
    Dataset_test = DiskSet_example(path, 'aug_data_example_test.txt')
    loader_test = torch.utils.data.DataLoader(dataset=Dataset_test,batch_size = 64, shuffle = False)
    device = torch.device("cuda:"+str(arg.cuda_id) if torch.cuda.is_available() else "cpu")
    #%%
    filename=[arg.net_name+'_disk_pca_regseg_P3b100_epoch99',
              arg.net_name+'_disk_pca_regseg_P5b100_epoch99',
              arg.net_name+'_disk_pca_regseg_P10b100_epoch99',
              #--------------------------------------------------------------------------------------------
              arg.net_name+'_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P3b100e100_ptm_epoch99',
              arg.net_name+'_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P5b100e100_ptm_epoch99',
              arg.net_name+'_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P10b100e100_ptm_epoch99',
              #--------------------------------------------------------------------------------------------
              arg.net_name+'_disk_pca_regseg_advreg_mae_0.07Linfi20s0.01_P3b100e100_ptm_epoch99',
              arg.net_name+'_disk_pca_regseg_advreg_mae_0.07Linfi20s0.01_P5b100e100_ptm_epoch99',
              arg.net_name+'_disk_pca_regseg_advreg_mae_0.07Linfi20s0.01_P10b100e100_ptm_epoch99',
              #--------------------------------------------------------------------------------------------
              arg.net_name+'_disk_pca_regseg_advseg_bce_dice_0.07Linfi20s0.01_P3b100e100_ptm_epoch99',
              arg.net_name+'_disk_pca_regseg_advseg_bce_dice_0.07Linfi20s0.01_P5b100e100_ptm_epoch99',
              arg.net_name+'_disk_pca_regseg_advseg_bce_dice_0.07Linfi20s0.01_P10b100e100_ptm_epoch99',
              #--------------------------------------------------------------------------------------------
              arg.net_name+'_disk_pca_regseg_rand_0.07Linf_P3b100e100_ptm_epoch99',
              arg.net_name+'_disk_pca_regseg_rand_0.07Linf_P5b100e100_ptm_epoch99',
              arg.net_name+'_disk_pca_regseg_rand_0.07Linf_P10b100e100_ptm_epoch99',
              ]
    #%%
    for m in range(0, len(filename)):
        try:
            if arg.net_name == 'Resnet18Unet':
                model = Resnet18Unet(352,1)
            model.output='seg'
            evaluate(model, device, 'result/'+filename[m], loader_test, arg.norm_type)
        except:
            print('error: result/'+filename[m])
    
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
from skimage.draw import polygon2mask
from Lumbar_Dataset import DiskSet
from Lumbar_Dataset import DiskSet_example
from Resnet18Unet import Resnet18Unet

from PGD import pgd_attack
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
def dice(Mp, M, reduction='none'):
    # NxKx128x128
    intersection = (Mp*M).sum(dim=(2,3))
    dice = (2*intersection) / (Mp.sum(dim=(2,3)) + M.sum(dim=(2,3)))
    if reduction == 'mean':
        dice = dice.mean()
    return dice
#%%
def poly_disk(S):
    device=S.device
    S=S.detach().cpu().numpy()
    Mask = np.zeros((S.shape[0],1,128,128), dtype=np.float32)
    for i in range(S.shape[0]):
        Mask[i,0]=polygon2mask((128,128), S[i])
        Mask[i,0]=np.transpose(Mask[i,0])
    Mask = torch.tensor(Mask, dtype=torch.float32, device=device)
    return Mask
#%%
def attack_loss(Sp, S):
    Sp=Sp.view(Sp.shape[0], -1)
    S=S.view(S.shape[0], -1)
    loss=((S-Sp)**2).sum()
    return loss
#%%
def test_adv(model, device, dataloader, noise_norm, norm_type, max_iter, step):
    model.eval()#set model to evaluation mode
    mrse_clean=[]
    mrse_noisy=[]
    dice_clean=[]
    dice_noisy=[]
    Xn_all=[]
    #---------------------
    for batch_idx, batch_data in enumerate(dataloader):        
        X, S = batch_data[0].to(device), batch_data[1].to(device)
        #------------------
        Sp = model(X)
        Xn = pgd_attack(model, X, S, noise_norm, norm_type, max_iter, step, use_optimizer=True, loss_fn=attack_loss)
        Spn = model(Xn)
        #------------------        
        batch_size=X.size(0)
        S=S.view(batch_size,-1, 2)
        Sp=Sp.view(batch_size,-1, 2)
        Spn=Spn.view(batch_size,-1, 2)
        M = poly_disk(S)
        Mp = poly_disk(Sp)
        Mpn = poly_disk(Spn)
        mrse_clean.append(((Sp-S)**2).sum(dim=2).sqrt().mean(dim=1).detach().cpu().numpy())
        mrse_noisy.append(((Spn-S)**2).sum(dim=2).sqrt().mean(dim=1).detach().cpu().numpy())
        dice_clean.append(dice(Mp, M).detach().cpu().numpy())
        dice_noisy.append(dice(Mpn, M).detach().cpu().numpy())    
        Xn_all.append(Xn.detach().cpu().numpy())
    mrse_clean=np.concatenate(mrse_clean)
    mrse_noisy=np.concatenate(mrse_noisy)
    dice_clean=np.concatenate(dice_clean)
    dice_noisy=np.concatenate(dice_noisy)
    Xn_all=np.concatenate(Xn_all)
    #------------------
    result={}
    result['noise_norm']=noise_norm
    result['mrse_clean']=mrse_clean
    result['mrse_noisy']=mrse_noisy
    result['dice_clean']=dice_clean
    result['dice_noisy']=dice_noisy
    result['Xn']=Xn_all
    #------------------
    print('test reg robustness, noise_norm:', noise_norm, 'norm_type', norm_type, 'max_iter', max_iter, 'step', step)
    print('mrse_clean: ', result['mrse_clean'].mean(), '(', result['mrse_clean'].std(), ')', sep='')
    print('mrse_noisy: ', result['mrse_noisy'].mean(), '(', result['mrse_noisy'].std(), ')', sep='')
    print('dice_clean: ', result['dice_clean'].mean(), '(', result['dice_clean'].std(), ')', sep='')
    print('dice_noisy: ', result['dice_noisy'].mean(), '(', result['dice_noisy'].std(), ')', sep='')
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
    filename_save=filename+'_evaluate_reg_L'+str(norm_type)
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
            model.output='reg'
            evaluate(model, device, 'result/'+filename[m], loader_test, arg.norm_type)
        except:
            print('error: result/'+filename[m])
    

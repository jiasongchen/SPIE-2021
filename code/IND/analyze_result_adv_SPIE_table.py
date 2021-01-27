# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Gu, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
namelist = [('Resnet18Unet_disk_pca_regseg_P3b100_epoch99',  'P03_std'),
            ('Resnet18Unet_disk_pca_regseg_P5b100_epoch99',  'p05_std'),
            ('Resnet18Unet_disk_pca_regseg_P10b100_epoch99', 'P10_std'),
            #--------------------------------------------------------------------------------------------
            ('Resnet18Unet_disk_pca_regseg_rand_0.07Linf_P3b100e100_ptm_epoch99',  'P03_rand'),
            ('Resnet18Unet_disk_pca_regseg_rand_0.07Linf_P5b100e100_ptm_epoch99',  'p05_rand'),
            ('Resnet18Unet_disk_pca_regseg_rand_0.07Linf_P10b100e100_ptm_epoch99',  'P10_rand'),
            #--------------------------------------------------------------------------------------------
            ('Resnet18Unet_disk_pca_regseg_advreg_mae_0.07Linfi20s0.01_P3b100e100_ptm_epoch99',  'P03_adv_r'),
            ('Resnet18Unet_disk_pca_regseg_advreg_mae_0.07Linfi20s0.01_P5b100e100_ptm_epoch99',  'P05_adv_r'),
            ('Resnet18Unet_disk_pca_regseg_advreg_mae_0.07Linfi20s0.01_P10b100e100_ptm_epoch99', 'P10_adv_r'),
            #--------------------------------------------------------------------------------------------
            ('Resnet18Unet_disk_pca_regseg_advseg_bce_dice_0.07Linfi20s0.01_P3b100e100_ptm_epoch99',  'P03_adv_s'),
            ('Resnet18Unet_disk_pca_regseg_advseg_bce_dice_0.07Linfi20s0.01_P5b100e100_ptm_epoch99',  'P05_adv_s'),
            ('Resnet18Unet_disk_pca_regseg_advseg_bce_dice_0.07Linfi20s0.01_P10b100e100_ptm_epoch99', 'P10_adv_s'),
            #--------------------------------------------------------------------------------------------
            ('Resnet18Unet_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P3b100e100_ptm_epoch99',  'P03_adv_rs'),
            ('Resnet18Unet_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P5b100e100_ptm_epoch99',  'P05_adv_rs'),
            ('Resnet18Unet_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P10b100e100_ptm_epoch99', 'P10_adv_rs'),
            #--------------------------------------------------------------------------------------------
            ]
#%%
noise=[0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2]
dfLinf_mrse=pd.DataFrame()
dfLinf_mrse['noise']=noise
dfLinf_dice_reg=pd.DataFrame()
dfLinf_dice_reg['noise']=noise
for m in range(0, len(namelist)):
    file='result/'+namelist[m][0]
    label=namelist[m][1]
    result_Linf=torch.load(file+'_evaluate_reg_Linf.pt')['result']
    #-----------
    mrse=[result_Linf[0]['mrse_clean'].mean()]
    for r in result_Linf:
        mrse.append(r['mrse_noisy'].mean())
    dfLinf_mrse[label]=mrse
    dice=[result_Linf[0]['dice_clean'].mean()]
    for r in result_Linf:
        dice.append(r['dice_noisy'].mean())
    dfLinf_dice_reg[label]=dice
#
dfLinf_mrse=dfLinf_mrse.transpose()#.round(decimals=4)
dfLinf_mrse.to_csv('result/SPIE/all_mrse.csv', header=False, float_format='%.4f')
dfLinf_dice_reg=dfLinf_dice_reg.transpose()#.round(decimals=4)
dfLinf_dice_reg.to_csv('result/SPIE/all_dice_reg.csv', header=False, float_format='%.4f')
#%%
dfLinf_dice_seg=pd.DataFrame()
dfLinf_dice_seg['noise']=noise
for m in range(0, len(namelist)):
    file='result/'+namelist[m][0]
    label=namelist[m][1]
    result_Linf=torch.load(file+'_evaluate_seg_Linf.pt')['result']
    dice=[result_Linf[0]['dice_clean'].mean()]
    for r in result_Linf:
        dice.append(r['dice_noisy'].mean())
    dfLinf_dice_seg[label]=dice
#plot    
dfLinf_dice_seg=dfLinf_dice_seg.transpose()#.round(decimals=4)
dfLinf_dice_seg.to_csv('result/SPIE/all_dice_seg.csv', header=False, float_format='%.4f')
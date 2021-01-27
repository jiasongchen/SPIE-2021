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
net='Resnet18Unet'
namelist=[]
tablename=[]
#%%
namelist0=[(net+'_disk_pca_regseg_P3b100_epoch99',  'P03_std', 'b', '--'),
           (net+'_disk_pca_regseg_P5b100_epoch99',  'P05_std', 'g', '--'),
           (net+'_disk_pca_regseg_P10b100_epoch99', 'P10_std', 'r', '--'),    
           (net+'_disk_pca_regseg_rand_0.07Linf_P3b100e100_ptm_epoch99',  'P03_rand', 'b', '-'),
           (net+'_disk_pca_regseg_rand_0.07Linf_P5b100e100_ptm_epoch99',  'P05_rand', 'g', '-'),
           (net+'_disk_pca_regseg_rand_0.07Linf_P10b100e100_ptm_epoch99', 'P10_rand', 'r', '-')
          ]
tablename0='rand_P3P5P10_0.07Linf'
namelist.append(namelist0)
tablename.append(tablename0)
#%%
namelist1=[(net+'_disk_pca_regseg_P3b100_epoch99',  'P03_std', 'b', '--'),
           (net+'_disk_pca_regseg_P5b100_epoch99',  'P05_std', 'g', '--'),
           (net+'_disk_pca_regseg_P10b100_epoch99', 'P10_std', 'r', '--'),    
           (net+'_disk_pca_regseg_advreg_mae_0.07Linfi20s0.01_P3b100e100_ptm_epoch99',  'P03_adv_r', 'b', '-'),
           (net+'_disk_pca_regseg_advreg_mae_0.07Linfi20s0.01_P5b100e100_ptm_epoch99',  'P05_adv_r', 'g', '-'),
           (net+'_disk_pca_regseg_advreg_mae_0.07Linfi20s0.01_P10b100e100_ptm_epoch99', 'P10_adv_r', 'r', '-')
          ]
tablename1='advreg_mae_P3P5P10_0.07Linf'
namelist.append(namelist1)
tablename.append(tablename1)
#%%
namelist2=[(net+'_disk_pca_regseg_P3b100_epoch99',  'P03_std', 'b', '--'),
           (net+'_disk_pca_regseg_P5b100_epoch99',  'P05_std', 'g', '--'),
           (net+'_disk_pca_regseg_P10b100_epoch99', 'P10_std', 'r', '--'),    
           (net+'_disk_pca_regseg_advseg_bce_dice_0.07Linfi20s0.01_P3b100e100_ptm_epoch99',  'P03_adv_s', 'b', '-'),
           (net+'_disk_pca_regseg_advseg_bce_dice_0.07Linfi20s0.01_P5b100e100_ptm_epoch99',  'P05_adv_s', 'g', '-'),
           (net+'_disk_pca_regseg_advseg_bce_dice_0.07Linfi20s0.01_P10b100e100_ptm_epoch99', 'P10_adv_s', 'r', '-')
          ]
tablename2='advseg_bce_dice_P3P5P10_0.07Linf'
namelist.append(namelist2)
tablename.append(tablename2)
#%%
namelist3=[(net+'_disk_pca_regseg_P3b100_epoch99', 'P03_std', 'b', '--'),
           (net+'_disk_pca_regseg_P5b100_epoch99', 'P05_std', 'g', '--'),
           (net+'_disk_pca_regseg_P10b100_epoch99', 'P10_std', 'r', '--'),    
           (net+'_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P3b100e100_ptm_epoch99',  'P03_adv_rs', 'b', '-'),
           (net+'_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P5b100e100_ptm_epoch99',  'P05_adv_rs', 'g', '-'),
           (net+'_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P10b100e100_ptm_epoch99', 'P10_adv_rs', 'r', '-')
          ]
tablename3='advregseg_mae_bce_dice_P3P5P10_0.07Linf'
namelist.append(namelist3)
tablename.append(tablename3)
#%%
idx=3
namelist=namelist[idx]
tablename=tablename[idx]
#
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
dfLinf_mrse=dfLinf_mrse.transpose().round(decimals=4)
dfLinf_mrse.to_csv('result/'+net+'_'+tablename+'_mrse.csv', header=False)
dfLinf_dice_reg=dfLinf_dice_reg.transpose().round(decimals=4)
dfLinf_dice_reg.to_csv('result/SPIE/'+net+'_'+tablename+'_dice_reg.csv', header=False)
#%%
xticks=np.linspace(0, noise[-1], len(noise))
fig, ax=plt.subplots(figsize=(5,4))
for m in range(0, len(namelist)):
    label=namelist[m][1]
    color=namelist[m][2]
    linestyle=namelist[m][3]
    ax.plot(xticks, dfLinf_mrse.loc[label].values, label=label, color=color, linestyle=linestyle)
ax.set_ylim(0, 26)
ax.set_yticks(np.linspace(0, 26, 14))
ax.set_xticks(xticks)
ax.set_xticklabels(noise)
ax.set_xlim(0, 0.2)
ax.set_xlabel('noise level (Linf)')
ax.set_title('shape regression error v.s. noise')
ax.grid(True)
ax.legend(loc='upper left', fontsize=10)
fig.savefig('result/SPIE/'+net+'_'+tablename+'_mrse.svg')
#%%
fig, ax=plt.subplots(figsize=(5,4))
for m in range(0, len(namelist)):
    label=namelist[m][1]
    color=namelist[m][2]
    linestyle=namelist[m][3]
    ax.plot(xticks, dfLinf_dice_reg.loc[label].values, label=label, color=color, linestyle=linestyle)    
ax.set_ylim(0.5, 1)
ax.set_yticks(np.linspace(0.5, 1, 11))
ax.set_xticks(xticks)
ax.set_xticklabels(noise)
ax.set_xlim(0, 0.2)
ax.set_xlabel('noise level (Linf)')
ax.set_title('shape regression accuracy v.s. noise')
ax.grid(True)
#ax.legend(loc='lower right')
fig.savefig('result/SPIE/'+net+'_'+tablename+'_dice_reg.svg')
#%%============================ segmentation ==============================================
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
dfLinf_dice_seg=dfLinf_dice_seg.transpose().round(decimals=4)
dfLinf_dice_seg.to_csv('result/SPIE/'+net+'_'+tablename+'_dice_seg.csv', header=False)
#
xticks=np.linspace(0, noise[-1], len(noise))
fig, ax=plt.subplots(figsize=(5,4))
for m in range(0, len(namelist)):
    label=namelist[m][1]
    color=namelist[m][2]
    linestyle=namelist[m][3]
    ax.plot(xticks, dfLinf_dice_seg.loc[label].values, label=label, color=color, linestyle=linestyle)   
ax.set_ylim(0.5, 1)
ax.set_yticks(np.linspace(0.5, 1, 11))
ax.set_xticks(xticks)
ax.set_xticklabels(noise)
ax.set_xlim(0, 0.2)
ax.set_xlabel('noise level (Linf)')
ax.set_title('image segmentation accuracy v.s. noise')
ax.grid(True)
#ax.legend(loc='lower right')
fig.savefig('result/SPIE/'+net+'_'+tablename+'_dice_seg.svg')
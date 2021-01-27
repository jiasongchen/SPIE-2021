# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Guc, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
#%%
from Lumbar_Dataset import DiskSet
from Resnet18Unet import Resnet18Unet, Resnet18a
from PGD import pgd_attack
from Disk_evaluate_reg import iou, dice, poly_disk
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(0)
np.random.seed(0)
import os
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(0)
#%%
def show_example_reg(model_list, model_name_list, idx, dataset, result_list, noise_list, figsize):
    fig, ax = plt.subplots(len(model_list), 7, figsize=figsize, constrained_layout=True)
    x=dataset[idx][0].reshape(1,1,128,128).to(device).contiguous()
    s=dataset[idx][1].reshape(1,-1,2).to(device).contiguous()
    ss=s.detach().cpu().numpy().squeeze()
    M = poly_disk(s)
    for m in range(0, len(model_list)):
        model=model_list[m]
        model.output='reg'
        result=result_list[m]
        for n in range(0, len(noise_list)):
            noise_norm = noise_list[n]
            if n >0:
                xn=torch.tensor(result[n-1]['Xn'][idx], device=device)      
            else:
                xn=x            
            temp=torch.norm(x-xn, p=norm_type)
            print('norm', norm_type, temp.item())
            xn=xn.view(1,1,128,128)
            sp=model(xn).view(1,-1, 2)
            Mp = poly_disk(sp)
            score=dice(Mp, M).detach().cpu().numpy().item()
            score='{:.3f}'.format(score)
            sp=sp.detach().cpu().numpy().squeeze()
            xn=xn.detach().cpu().numpy().reshape(128, 128)   
            ax[m,n].imshow(xn, cmap='gray', vmin=0, vmax=1)
            ax[m,n].plot(ss[:,0], ss[:,1], '-r', linewidth=3)
            ax[m,n].plot(sp[:,0], sp[:,1], '-g', linewidth=3)
            ax[m,n].set_xticks([])
            ax[m,n].set_yticks([])
            ax[m,n].set_title('ε='+str(noise_norm) +', dice='+score, fontsize=12)
        ax[m,0].set_ylabel(model_name_list[m], fontsize=16, labelpad=10)
    return fig, ax
#%%
from skimage.measure import find_contours
def show_example_seg(model_list, model_name_list, idx, dataset, result_list, noise_list, figsize):
    fig, ax = plt.subplots(len(model_list), 7, figsize=figsize, constrained_layout=True)
    x=dataset[idx][0].reshape(1,1,128,128).to(device).contiguous()
    s=dataset[idx][1].reshape(1,-1,2).to(device).contiguous()
    ss=s.detach().cpu().numpy()
    M = poly_disk(s)
    for m in range(0, len(model_list)):
        model=model_list[m]
        model.output='seg'
        result=result_list[m]
        for n in range(0, len(noise_list)):
            noise_norm = noise_list[n]
            if n >0:
                xn=torch.tensor(result[n-1]['Xn'][idx], device=device)      
            else:
                xn=x            
            temp=torch.norm(x-xn, p=norm_type)
            print('norm', norm_type, temp.item())
            xn=xn.view(1,1,128,128)
            Mp=model(xn)
            Mp=(Mp>0).to(torch.float32)
            score=dice(Mp, M).detach().cpu().numpy().item()
            score='{:.3f}'.format(score)
            xn=xn.detach().cpu().numpy().reshape(128, 128)   
            Mp=Mp.detach().cpu().reshape(128, 128)            
            sp=find_contours(Mp, level=0.5)      
            ax[m,n].imshow(xn, cmap='gray', vmin=0, vmax=1)
            ax[m,n].plot(ss[0,:,0], ss[0,:,1], '-r', linewidth=3)
            for ssp in sp:
                ax[m,n].plot(ssp[:,1], ssp[:,0], '-g', linewidth=3)
            ax[m,n].set_xticks([])
            ax[m,n].set_yticks([])
            ax[m,n].set_title('ε='+str(noise_norm) +', dice='+score, fontsize=12)
        ax[m,0].set_ylabel(model_name_list[m], fontsize=16, labelpad=10)
    return fig, ax
#%%
if __name__ == '__main__':
    #%%
    path='../../data/Lumbar/UM100_disk/'
    Dataset_test = DiskSet(path, '100_test.txt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #%%
    filename=['Resnet18Unet_disk_pca_regseg_P10b100_epoch99',
              'Resnet18Unet_disk_pca_regseg_rand_0.07Linf_P10b100e100_ptm_epoch99',
              'Resnet18Unet_disk_pca_regseg_advregseg_mae_bce_dice_0.01Linfi20s0.01_P10b100e100_ptm_epoch99']
    model_name_list=['P10_std', 'P10_rand', 'P10_adv_rs']
    model_list=[]
    result_list_reg=[]
    result_list_seg=[]
    for m in range(0, len(filename)):
        model = Resnet18Unet(352,1)
        checkpoint=torch.load('result/'+filename[m]+'.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device);
        model.eval();
        model_list.append(model)
        result_list_reg.append(torch.load('result/'+filename[m]+'_evaluate_reg_Linf.pt')['result'])
        result_list_seg.append(torch.load('result/'+filename[m]+'_evaluate_seg_Linf.pt')['result'])
    #%%
    noise_list=[0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2]
    for i in range(0, len(Dataset_test)):
        for norm_type in [np.inf]:
            fig1, ax1 = show_example_reg(model_list, model_name_list, i, Dataset_test,  result_list_reg,  noise_list, figsize=(16, 8))
            fig1.savefig('result/SPIE/figure_Linf_compare3/'+str(i)+'_reg.png')
            plt.close(fig1)
            fig2, ax2 = show_example_seg(model_list, model_name_list, i, Dataset_test,  result_list_seg,  noise_list, figsize=(16, 8))
            fig2.savefig('result/SPIE/figure_Linf_compare3/'+str(i)+'_seg.png')
            plt.close(fig2)
            print('i=', i)
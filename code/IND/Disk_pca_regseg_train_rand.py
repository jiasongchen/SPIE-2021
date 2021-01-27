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
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.optim import Adamax
from tqdm import tqdm
import argparse
from Lumbar_Dataset import DiskSet
from Lumbar_Dataset import DiskSet_example
from Resnet18Unet import Resnet18Unet
from PCA_Aug import PCA_Aug_Dataloader
from Disk_regseg_train import dice, poly_disk, test, plot_history, save_checkpoint, load_checkpoint
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(0)
#%%
def train(model, device, optimizer, dataloader, epoch, arg):
    noise_norm=arg.noise_norm
    epoch_refine=arg.epoch_refine
    #-----------------
    model.train()
    loss1_train=0
    loss2_train=0
    loss3_train=0
    noise_norm=min(noise_norm, noise_norm*(epoch+1)/epoch_refine)
    for batch_idx, (X, S) in enumerate(dataloader):
        X, S = X.to(device), S.to(device)
        M=poly_disk(S)
        model.output='reg_seg'
        Sp, Mp=model(X)
        Xn=noise_norm*torch.rand_like(X)
        Spn, Mpn=model(Xn)
        loss_ce=(nnF.binary_cross_entropy_with_logits(Mp, M)+nnF.binary_cross_entropy_with_logits(Mpn, M))/2
        Mp=torch.sigmoid(Mp)
        Mpn=torch.sigmoid(Mpn)
        loss_dice=((1-dice(Mp, M, 'mean'))+(1-dice(Mpn, M, 'mean')))/2
        Sp=Sp.view(Sp.shape[0], -1, 2)
        Spn=Spn.view(Spn.shape[0], -1, 2)
        loss_mae= ((S-Sp).abs().mean()+(S-Spn).abs().mean())/2
        loss=(loss_ce+loss_dice)/2+loss_mae
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1_train+=loss_ce.item()
        loss2_train+=loss_dice.item()
        loss3_train+=loss_mae.item()
    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    loss3_train/=len(dataloader)
    return loss1_train, loss2_train, loss3_train
#%%
if __name__ == '__main__':
    #%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--net_name', default='Resnet18Unet', type=str)
    parser.add_argument('--norm_type', default=np.inf, type=float)
    parser.add_argument('--noise_norm', default=0.07, type=float)
    parser.add_argument('--n_components', default=10, type=int)
    parser.add_argument('--n_batches', default=100, type=int)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epoch_refine', default=100, type=int)
    parser.add_argument('--epoch_end', default=100, type=int)
    parser.add_argument('--ptm', default=0, type=int)#1: use pre-trained model
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--path', default='../data/', type=str)
    parser.add_argument('--path_aug', default='../data/', type=str)
    arg = parser.parse_args()
    print(arg)
    device = torch.device("cuda:"+str(arg.cuda_id) if torch.cuda.is_available() else "cpu")
    #%%
    loader_train=PCA_Aug_Dataloader(n_epochs=100, n_batches=arg.n_batches, batch_size=64, device=device, shuffle=True,
                                    filename=arg.path_aug+'pca_aug_P'+str(arg.n_components)+'b'+str(arg.n_batches)+'n64',
                                    n_components=arg.n_components, c_max=2, flag=0, train=True, path=arg.path)
    loader_val=PCA_Aug_Dataloader(n_epochs=1, n_batches=100, batch_size=64, device=device, shuffle=False,
                                  filename=arg.path_aug+'pca_aug_P30b100n64',
                                  n_components=30, c_max=2, flag=0, train=True, path=arg.path)
    Dataset_test = DiskSet_example(arg.path, 'aug_data_example_test.txt')
    loader_test = torch.utils.data.DataLoader(dataset=Dataset_test,batch_size = 64, shuffle = False, num_workers=0)
    #%%
    filename=('result/'+arg.net_name+'_disk_pca_regseg_rand'
              +'_'+str(arg.noise_norm)+'L'+str(arg.norm_type)
              +'_P'+str(arg.n_components)+'b'+str(arg.n_batches)+'e'+str(arg.epoch_refine))
    if arg.ptm == 1:
        filename+='_ptm'
    print('save to', filename)
    #%%
    if arg.net_name == 'Resnet18Unet':
        model = Resnet18Unet(352, 1)
    if arg.ptm == 1:
        filename_pre=('result/'+arg.net_name+'_disk_pca_regseg'
                      +'_P'+str(arg.n_components)+'b'+str(arg.n_batches)+'_epoch99.pt')
        load_checkpoint(filename_pre, model, None, None)
    model.to(device)
    #%%
    optimizer = Adamax(model.parameters(),lr = 0.001)
    history={'loss1_train':[], 'loss2_train':[], 'loss3_train':[],
             'mrse_val':[], 'dice1_val':[], 'dice2_val':[],
             'mrse_test':[], 'dice1_test':[], 'dice2_test':[]}
    #%% load model state and optimizer state if necessary
    epoch_save=arg.epoch_start-1
    if epoch_save>=0:
        load_checkpoint(filename+'_epoch'+str(epoch_save)+'.pt', model, optimizer, history)
    #%%
    for epoch in tqdm(range(epoch_save+1, arg.epoch_end), initial=epoch_save+1, total=arg.epoch_end):
        loss_train = train(model, device, optimizer, loader_train, epoch, arg)
        mrse_val, dice1_val, dice2_val = test(model, device, loader_val)
        mrse_test, dice1_test, dice2_test = test(model, device, loader_test)
        history['loss1_train'].append(loss_train[0])
        history['loss2_train'].append(loss_train[1])
        history['loss3_train'].append(loss_train[2])
        history['mrse_val'].append(mrse_val.mean())
        history['dice1_val'].append(dice1_val.mean())
        history['dice2_val'].append(dice2_val.mean())     
        history['mrse_test'].append(mrse_test.mean())
        history['dice1_test'].append(dice1_test.mean())
        history['dice2_test'].append(dice2_test.mean())  
        #------- show result ----------------------
        display.clear_output(wait=False)
        fig1, ax1 = plot_history(history)            
        display.display(fig1)        
        fig2, ax2 = plt.subplots()
        ax2.hist(mrse_test, bins=50, range=(0,10))
        ax2.set_xlim(0, 10)
        display.display(fig2)
        #----------save----------------------------
        if (epoch+1)%1== 0:
            save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, history, epoch)
            fig1.savefig(filename+'_epoch'+str(epoch)+'_history.png')
            fig2.savefig(filename+'_epoch'+str(epoch)+'_mrse_test.png')
        epoch_save=epoch
        plt.close(fig1)
        plt.close(fig2)

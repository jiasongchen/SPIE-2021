# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Gu, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""
import os 
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn.functional as nnF
from torch.optim import Adamax
from tqdm import tqdm
import argparse
from skimage.draw import polygon2mask
from Lumbar_Dataset import DiskSet
from Lumbar_Dataset import DiskSet_example
from Resnet18Unet import Resnet18Unet
from PCA_Aug import PCA_Aug_Dataloader
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
def dice(Mp, M, reduction='none'):
    # NxKx128x128
    intersection = (Mp*M).sum(dim=(2,3))
    dice = (2*intersection) / (Mp.sum(dim=(2,3)) + M.sum(dim=(2,3)))
    if reduction == 'mean':
        dice = dice.mean()
    elif reduction == 'sum':
        dice = dice.sum()
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
def dice_shape(Sp, S):
    S=S.view(S.shape[0], -1, 2)
    Sp=Sp.view(Sp.shape[0], -1, 2)
    M = poly_disk(S)
    Mp = poly_disk(Sp)
    score = dice(Mp, M)
    return score
#%%
def mrse_shape(Sp, S):
    S=S.view(S.shape[0], -1, 2)
    Sp=Sp.view(Sp.shape[0], -1, 2)
    error = ((Sp-S)**2).sum(dim=2).sqrt().mean(dim=1)
    return error
#%%
def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr']=new_lr
        print('new lr=', g['lr'])
#%%
def train(model, device, optimizer, dataloader, epoch):
    model.train()
    model.output='reg_seg'
    loss1_train=0
    loss2_train=0
    loss3_train=0
    for batch_idx, (X, S) in enumerate(dataloader):
        X, S = X.to(device), S.to(device)
        M=poly_disk(S)
        Sp, Mp=model(X)
        loss_ce=nnF.binary_cross_entropy_with_logits(Mp, M)               
        Mp=torch.sigmoid(Mp)
        loss_dice=1-dice(Mp, M, 'mean')
        #--------------------------------------
        Sp=Sp.view(Sp.shape[0], -1, 2)
        loss_mae=(S-Sp).abs().mean()        
        #--------------------------------------
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
def test(model, device, dataloader):
    model.eval()
    model.output='reg_seg'
    mrse_test=[]
    dice1_test=[]
    dice2_test=[]
    with torch.no_grad():
        for batch_idx, (X, S) in enumerate(dataloader):
            M=poly_disk(S).to(device)
            X, S = X.to(torch.float32).to(device), S.to(torch.float32).to(device)            
            Sp, Mp=model(X)
            Sp=Sp.view(Sp.shape[0], -1, 2)
            temp= ((Sp-S)**2).sum(dim=2).sqrt().mean(dim=1)
            mrse_test.append(temp.detach().cpu().numpy())
            Mpp = poly_disk(Sp)
            temp=dice(Mpp, M)
            dice1_test.append(temp.detach().cpu().numpy())            
            Mp=Mp>0
            temp=dice(Mp, M)
            dice2_test.append(temp.detach().cpu().numpy())        
    #------------------
    mrse_test=np.concatenate(mrse_test)
    dice1_test=np.concatenate(dice1_test)
    dice2_test=np.concatenate(dice2_test)
    return mrse_test, dice1_test, dice2_test
#%%
def plot_history(history):
    fig, ax = plt.subplots(2, 3, constrained_layout=True)
    ax[0,0].plot(history['loss3_train'], '-b', label='loss3_train')
    ax[0,0].set_ylim(0, 3)
    ax[0,0].set_yticks(np.linspace(0, 3, 7))
    ax[0,0].grid(True)
    ax[0,0].legend()
    
    ax[0,1].plot(history['mrse_val'], '-r', label='mrse_val')
    ax[0,1].set_ylim(0, 3)
    ax[0,1].set_yticks(np.linspace(0, 3, 7))
    ax[0,1].grid(True)
    ax[0,1].legend() 

    ax[0,2].plot(history['mrse_test'], '-r', label='mrse_test')
    ax[0,2].set_ylim(0, 3)
    ax[0,2].set_yticks(np.linspace(0, 3, 7))
    ax[0,2].grid(True)
    ax[0,2].legend() 
    
    ax[1,0].plot(history['loss2_train'], '-b', label='loss2_train')
    ax[1,0].plot(history['loss1_train'], '-c', label='loss1_train')
    ax[1,0].set_ylim(0, 0.1)
    ax[1,0].grid(True)
    ax[1,0].legend()
        
    ax[1,1].plot(history['dice1_val'], '-r', label='dice1_val')
    ax[1,1].plot(history['dice2_val'], '-c', label='dice2_val')
    ax[1,1].set_ylim(0.84, 1)
    ax[1,1].set_yticks(np.linspace(0.84, 1, 9))
    ax[1,1].grid(True)
    ax[1,1].legend()

    ax[1,2].plot(history['dice1_test'], '-r', label='dice1_test')
    ax[1,2].plot(history['dice2_test'], '-c', label='dice2_test')
    ax[1,2].set_ylim(0.84, 1)
    ax[1,2].set_yticks(np.linspace(0.84, 1, 9))
    ax[1,2].grid(True)
    ax[1,2].legend()
      
    return fig, ax   
#%%
def save_checkpoint(filename, model, optimizer, history, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history':history},
               filename)
    print('saved:', filename)
#%%
def load_checkpoint(filename, model, optimizer, history):
    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    if history is not None:
        if 'history' in checkpoint.keys():
            history.update(checkpoint['history'])
        else:
            history.update(checkpoint['result'])
    print('loaded:', filename)
#%%
if __name__ == '__main__':
    #%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--net_name', default='Resnet18Unet', type=str)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epoch_end', default=1000, type=int)
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--path', default='../data/', type=str)
    parser.add_argument('--path_aug', default='../data/', type=str)
    
    
    arg = parser.parse_args()
    print(arg)
    device = torch.device("cuda:"+str(arg.cuda_id) if torch.cuda.is_available() else "cpu")
    #%%
    Dataset_train = DiskSet_example(arg.path, 'aug_data_example_train.txt')
    Dataset_test = DiskSet_example(arg.path, 'aug_data_example_test.txt')
    loader_train = torch.utils.data.DataLoader(dataset=Dataset_train,batch_size = 64, shuffle = True, num_workers=0)
    loader_test = torch.utils.data.DataLoader(dataset=Dataset_test,batch_size = 64, shuffle = False, num_workers=0)
    #%% validation using pca_aug data on training set
    loader_val=PCA_Aug_Dataloader(n_epochs=1, n_batches=100, batch_size=64, device=device, shuffle = False,
                                  filename=arg.path_aug+'pca_aug_P30b100n64',
                                  n_components=30, c_max=2, flag=0, train=True, path=arg.path)
    #%%
    filename='result/'+arg.net_name+'_disk_regseg'
    print('save to', filename)
    #%%
    if arg.net_name == 'Resnet18Unet':
        model = Resnet18Unet(352, 1).to(device)
    optimizer = Adamax(model.parameters(),lr = 0.001)
    history={'loss1_train':[], 'loss2_train':[], 'loss3_train':[],
             'mrse_val':[], 'dice1_val':[], 'dice2_val':[],
             'mrse_test':[], 'dice1_test':[], 'dice2_test':[]}
    #%%
    #print(torch.cuda.memory_summary(device=device, abbreviated=True))
    #%% load model state and optimizer state if necessary
    epoch_save=arg.epoch_start-1
    if epoch_save>=0:
        load_checkpoint(filename+'_epoch'+str(epoch_save)+'.pt', model, optimizer, history)
    #%%
    for epoch in tqdm(range(epoch_save+1, arg.epoch_end), initial=epoch_save+1, total=arg.epoch_end):
        loss_train = train(model, device, optimizer, loader_train, epoch)
        #print(torch.cuda.memory_summary(device=device, abbreviated=True))
        mrse_val, dice1_val, dice2_val = test(model, device, loader_train)
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
        if (epoch+1)%1000 == 0:
            save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, history, epoch)
            fig1.savefig(filename+'_epoch'+str(epoch)+'_history.png')
            fig2.savefig(filename+'_epoch'+str(epoch)+'_mrse_test.png')
        epoch_save=epoch
        plt.close(fig1)
        plt.close(fig2)

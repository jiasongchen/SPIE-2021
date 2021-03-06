# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Gu, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""
import sys
sys.path.append('../../core')
import sys
sys.path.append('../')
#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as nnF
import skimage
import skimage.io as io
import skimage.transform as sk_transform
import argparse
from PGD_OOD import pgd_attack
from Lumbar_Dataset import DiskSet
from Lumbar_Dataset import DiskSetexample
from Resnet18Unet import Resnet18Unet
from Disk_regseg_train import dice, dice_shape
#%%
def load_dataset(data_path, batch_size, num_workers):    
    Dataset_test = DiskSetexample(data_path, 'aug_data_example_test.txt')
    loader_test = torch.utils.data.DataLoader(dataset=Dataset_test,batch_size = batch_size,
                                              shuffle = False, num_workers=num_workers)    
    return loader_test
#%%
loss_record=[]
def loss_model(model, X1, Y, X2):
    model.output='reg_seg'
    Sp1, Mp1=model(X1)
    Mp1=torch.sigmoid(Mp1)
    Sp1=Sp1.view(Sp1.shape[0], -1)
    
    Sp2, Mp2=model(X2)
    Mp2=torch.sigmoid(Mp2)
    Sp2=Sp2.view(Sp2.shape[0], -1)
    
    loss_mse=((Sp2-Sp1)**2).mean()    
    loss_dice=1-dice(Mp2, Mp1)
    loss=loss_mse+loss_dice

    if loss_record is not None:
        loss_record.append(loss.item())
    return loss
#---------------------------------------------------------------------------
def attack(model, x1, x2, noise_norm, norm_type, max_iter, step):
    xn=pgd_attack(model, x1, None, x2,
                  noise_norm=noise_norm, norm_type=norm_type,
                  max_iter=max_iter, step=step, use_optimizer=True, loss_model=loss_model)
    return xn       
#%%
if __name__ == '__main__':    
    #%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--net_name', default='Resnet18Unet', type=str)
    parser.add_argument('--ood_name', default='ct', type=str)
    parser.add_argument('--path', default='../data/', type=str)
    #-------------------------------------------
    arg = parser.parse_args()
    print(arg)
    device=torch.device('cuda:'+str(arg.cuda_id) if torch.cuda.is_available() else "cpu")
    net_name=arg.net_name
    ood_name=arg.ood_name
    #-------------------------------------------
    if net_name == 'Resnet18Unet':
        model = Resnet18Unet(352, 1)
        filename_pre='result/Resnet18Unet_disk_pca_regseg_P10b100_epoch99.pt'
        model.load_state_dict(torch.load(filename_pre, map_location='cpu')['model_state_dict'])
    elif net_name == 'Resnet18Unet_advregseg':
        model = Resnet18Unet(352, 1)
        filename_pre='result/Resnet18Unet_disk_pca_regseg_advregseg_mae_bce_dice_0.07Linfi20s0.01_P10b100e100_ptm_epoch99.pt'
        model.load_state_dict(torch.load(filename_pre, map_location='cpu')['model_state_dict'])
    model.eval()
    model.to(device)
    #%%
    loader=load_dataset(arg.path, 1, 0)
    #%% 78 , 36
    idx1=1
    x1=loader.dataset[idx1][0]
    x1=torch.tensor(x1.reshape(1,1,128,128), device=device).contiguous()    
    xx1=x1.detach().cpu().numpy().reshape(128,128)
    plt.imshow(xx1, cmap='gray')
    #%% OOD: rand
    if ood_name == 'rand':
        x2=np.random.rand(1,128,128).astype('float32')
    elif ood_name == 'box':
        x2=np.random.rand(16,16).astype('float32')
        x2=sk_transform.resize(x2, [128, 128], order=0)
    elif ood_name == 'ct' or ood_name == 'x_ray':
        x2=io.imread('data/'+ood_name+'.jpg')
        x2=skimage.util.img_as_float32(x2)
        if len(x2.shape)>=3:
            x2=x2[:,:,0]
        x2=sk_transform.resize(x2, [128, 128])
    else:
        x2=loader.dataset[60][0]
    x2=torch.tensor(x2.reshape(1,1,128,128), device=device).contiguous()    
    xx2=x2.detach().cpu().numpy().reshape(128,128)
    plt.imshow(xx2, cmap='gray')
    #%%
    #'''
    start = time.time()
    noise_norm=20
    norm_type=2
    max_iter=2000
    step=0.01
    loss_record.clear()
    xn=attack(model, x1, x2, noise_norm, norm_type, max_iter, step)
    end = time.time()
    print('time cost:', end - start)
    print('norm:', torch.norm(xn-x2, p=norm_type).item())
    fig, ax = plt.subplots()
    ax.plot(loss_record)
    ax.set_ylim(0, 1)
    ax.grid(True)
    #---------------------------------------------------------------------
    zna, znb=model(xn)
    z1a, z1b=model(x1)
    z2a, z2b=model(x2)
    z1a=z1a.view(z1a.shape[0], -1, 2)
    z2a=z2a.view(z2a.shape[0], -1, 2)
    zna=zna.view(zna.shape[0], -1, 2)
    z1b=(z1b>0).to(torch.float32)
    z2b=(z2b>0).to(torch.float32)
    znb=(znb>0).to(torch.float32)

    dice_shape_x2= dice_shape(z2a, z1a).item()
    dice_seg_x2= dice(z2b, z1b).item()
    dice_shape_xn= dice_shape(zna, z1a).item()
    dice_seg_xn= dice(znb, z1b).item()
    print('dice_shape_x2', dice_shape_x2, 'dice_seg_x2', dice_seg_x2)
    print('dice_shape_xn', dice_shape_xn, 'dice_seg_xn', dice_seg_xn)

    z1a=z1a.squeeze().detach().cpu().numpy()
    z2a=z2a.squeeze().detach().cpu().numpy()
    zna=zna.squeeze().detach().cpu().numpy()
    z1b=z1b.squeeze().detach().cpu().numpy()
    z2b=z2b.squeeze().detach().cpu().numpy()
    znb=znb.squeeze().detach().cpu().numpy()        
    #----------------------------------------------------------
    fig, ax =plt.subplots(2, 3, sharex=True, sharey=True, constrained_layout=True, figsize=(10,6))
    ax[0,0].set_title('$x_1$', fontsize=16)
    ax[0,0].plot(z1a[:,0], z1a[:,1], '-b')  
    ax[0,0].imshow(x1.squeeze().detach().cpu().numpy(), cmap='gray')
    ax[1,0].plot(z1a[:,0], z1a[:,1], '--b')
    ax[1,0].axis([0, 127, 0, 127])
    ax[1,0].set_aspect(1)
    ax[1,0].imshow(z1b, cmap='gray')

    ax[0,1].set_title('$x_2$', fontsize=16)
    #ax[0,1].plot(z2a[:,0], z2a[:,1], '-m')
    ax[0,1].imshow(x2.squeeze().detach().cpu().numpy(), cmap='gray')
    ax[1,1].set_title('dice_shape='+'{:.3f}'.format(dice_shape_x2)+' dice_seg='+'{:.3f}'.format(dice_seg_x2))
    ax[1,1].plot(z2a[:,0], z2a[:,1], '-m')
    ax[1,1].plot(z1a[:,0], z1a[:,1], '--b')
    ax[1,1].axis([0, 127, 0, 127])
    ax[1,1].set_aspect(1)
    ax[1,1].imshow(z2b, cmap='gray')
    
    ax[0,2].set_title('$x_2$+\u03B4', fontsize=16)
    #ax[0,2].plot(zna[:,0], zna[:,1], '-m')
    ax[0,2].imshow(xn.squeeze().detach().cpu().numpy(), cmap='gray')
    ax[1,2].set_title('dice_shape='+'{:.3f}'.format(dice_shape_xn)+' dice_seg='+'{:.3f}'.format(dice_seg_xn))
    ax[1,2].plot(zna[:,0], zna[:,1], '-m')
    ax[1,2].plot(z1a[:,0], z1a[:,1], '--b')
    ax[1,2].axis([0, 127, 0, 127])
    ax[1,2].set_aspect(1)
    ax[1,2].imshow(znb, cmap='gray')
    #%%
    torch.backends.cudnn.benchmark = True
    #------------------------------------
    if ood_name == 'rand': 
        noise_norm=20
        norm_type=2
        max_iter=2000
        step=0.01 
    elif ood_name == 'ct': 
        noise_norm=20
        norm_type=2
        max_iter=2000
        step=0.01
    else:
        raise ValueError('unkown ood_name')
    #-----------------------------  
    filename=('result/'+net_name+'_OOD_regseg_'+str(noise_norm)+'L'+str(norm_type)
               +'_i'+str(max_iter)+'s'+str(step)
               +'_'+ood_name)
    print('save to', filename)
    print('batch:', len(loader))
    loss_record=None
    X2_list=[] 
    dice_shape_out=[]
    dice_seg_out=[]
    for batch_idx, (X1, target) in enumerate(loader):
        start = time.time()
        print(batch_idx)
        X1=X1.to(device)
        if ood_name != 'rand':
            X2=x2.expand(X1.shape[0],1,128,128).to(device)
        else:
            X2=torch.rand_like(X1)
        X2=attack(model, X1, X2, noise_norm, norm_type, max_iter, step)
        Z1a, Z1b=model(X1)
        Z2a, Z2b=model(X2)
        Z1b=(Z1b>0).to(torch.float32)
        Z2b=(Z2b>0).to(torch.float32)
        dice_shape_out.append(dice_shape(Z2a, Z1a).detach().cpu().numpy())        
        dice_seg_out.append(dice(Z2b, Z1b).detach().cpu().numpy())
        X2_list.append(X2.detach().cpu().numpy())
        end = time.time()
        print('time cost:', end - start)
        print('dice_shape_out', dice_shape_out[-1], 'dice_seg_out', dice_seg_out[-1])
        del X1, X2, Z1a, Z1b, Z2a, Z2b        
    dice_seg_out=np.concatenate(dice_seg_out)
    dice_shape_out=np.concatenate(dice_shape_out)
    X2_list=np.concatenate(X2_list) 
    #-------------------------------------------------
    torch.save({'dice_seg_out':dice_seg_out, 
                'dice_shape_out':dice_shape_out, 
                'X2_list':X2_list}, filename+'.pt')
#%%
if 0:
    #%%
    filename='Resnet18Unet_OOD_regseg_20L2_i2000s0.01_ct'
    #filename='Resnet18Unet_OOD_regseg_20L2_i2000s0.01_rand'
    #filename='Resnet18Unet_advregseg_OOD_regseg_20L2_i2000s0.01_rand'
    #filename='Resnet18Unet_advregseg_OOD_regseg_20L2_i2000s0.01_ct'
    result=torch.load('result/'+filename+'.pt')
    X2_list=result['X2_list']
    for i in range(X2_list.shape[0]):
        xi=X2_list[i].transpose(1,2,0)
        xi=255*(xi-xi.min())/(xi.max()-xi.min())
        xi=xi.astype('uint8')
        io.imsave('result/'+filename+'/x2_'+str(i)+'.png',  xi)
    #%%
    fig, ax = plt.subplots(1,2)
    ax[0].hist(result['dice_shape_out'])
    ax[1].hist(result['dice_seg_out'])
    #ax[1].set_xlim(0.99, 1)
    #%%
    dice_shape_out=[]
    dice_seg_out=[]
    for idx, (x1, target) in enumerate(loader.dataset):
        x1=x1.to(device)
        x2= torch.tensor(X2_list[idx], device=device)
        z1a, z1b=model(x1.view(1,1,128,128))
        z2a, z2b=model(x2.view(1,1,128,128))
        z1a=z1a.view(z1a.shape[0], -1, 2)
        z2a=z2a.view(z2a.shape[0], -1, 2)
        z1b=(z1b>0).to(torch.float32)
        z2b=(z2b>0).to(torch.float32)
        dice_shape_x2= dice_shape(z2a, z1a).item()
        dice_seg_x2= dice(z2b, z1b).item()
        
        dice_shape_out.append(dice_shape_x2)
        dice_seg_out.append(dice_seg_x2)
        
        z1a=z1a.squeeze().detach().cpu().numpy()
        z2a=z2a.squeeze().detach().cpu().numpy()    
        z1b=z1b.squeeze().detach().cpu().numpy()
        z2b=z2b.squeeze().detach().cpu().numpy()        
        #
        fig, ax =plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(7,7))
        ax[0,0].set_title('$x_{in}$', fontsize=16)
        ax[0,0].plot(z1a[:,0], z1a[:,1], '-g', linewidth=3)  
        ax[0,0].imshow(x1.squeeze().detach().cpu().numpy(), cmap='gray')
        ax[1,0].set_title('(a1, b1)=model($x_{in}$)', fontsize=16)
        ax[1,0].text(5, 115, 'a1: shape regression', color='w', fontsize=16)
        ax[1,0].text(5, 100, 'b1: image segmentation', color='w', fontsize=16)   
        ax[1,0].plot(z1a[:,0], z1a[:,1], '-g', linewidth=3)  
        ax[1,0].axis([0, 127, 0, 127])
        ax[1,0].set_aspect(1)
        ax[1,0].imshow(z1b, cmap='gray')

        ax[0,1].set_title('$x_{out}$', fontsize=16)
        #ax[0,1].plot(z2a[:,0], z2a[:,1], '-b', linewidth=3)
        ax[0,1].imshow(x2.squeeze().detach().cpu().numpy(), cmap='gray')
        ax[1,1].set_title('(a2, b2)=model($x_{out}$)', fontsize=16)
        ax[1,1].text(5, 115, 'a2: shape regression', color='w', fontsize=16)
        ax[1,1].text(5, 100, 'b2: image segmentation', color='w', fontsize=16)   
        ax[1,1].text(5, 15, 'dice(a1, a2)={:.3f}'.format(dice_shape_x2), color='w', fontsize=16)
        ax[1,1].text(5, 5,  'dice(b1, b2)={:.3f}'.format(dice_seg_x2), color='w', fontsize=16)        
        ax[1,1].plot(z2a[:,0], z2a[:,1], '-b', linewidth=3)
        ax[1,1].axis([0, 127, 0, 127])
        ax[1,1].set_aspect(1)
        ax[1,1].imshow(z2b, cmap='gray')
        
        fig.savefig('result/'+filename+'/x_in_out_'+str(idx)+'.png') 
        fig.savefig('result/'+filename+'/x_in_out_'+str(idx)+'.svg') 
        plt.close(fig)
        print(idx)
        #break
    #%%
    fig, ax = plt.subplots(1,2, constrained_layout=True)
    ax[0].hist(dice_shape_out, bins=20, range=(0.95, 1))
    ax[0].set_xticks(np.linspace(0.95, 1, 6))
    ax[0].set_xlim(0.95, 1)
    ax[0].set_title('dice histogram (regression)')
    ax[1].hist(dice_seg_out, bins=20, range=(0.95, 1))
    ax[1].set_xticks(np.linspace(0.95, 1, 6))
    ax[1].set_xlim(0.95, 1)
    ax[1].set_title('dice histogram (segmentation)')
    fig.savefig('result/'+filename+'/x_in_out_hist.svg') 
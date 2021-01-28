# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Gu, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""
#%%
import torch
from torch import optim
from PGD import get_noise_init, normalize_grad_, clip_norm_
#%%
def run_model_(model, X):
    Z=model(X)
    return Z
#%%
def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr']=new_lr
#%%
def pgd_attack(model, Xin, Y, Xout, noise_norm, norm_type, max_iter, step,
               rand_init=True, rand_init_norm=None, rand_init_Xn=None,
               clip_X_min=0, clip_X_max=1,
               clip_E_min=-1, clip_E_max=1,
               use_optimizer=False, loss_fn=None, run_model=None, loss_model=None,
               lr_schedule={}):
    # Xin:  IND sample-batch
    # Xout: OOD sample-batch
    #-----------------------------------------------------
    model.eval()#set model to evaluation mode
    #-----------------------------------------------------
    if loss_fn is None:
        loss_fn=torch.nn.MSELoss()
    #-----------------------------------------------------
    if run_model is None:
        run_model=run_model_    
    #-----------------
    if rand_init == True:
        if rand_init_Xn is not None:
            Xn=torch.clamp(rand_init_Xn.detach(), clip_X_min, clip_X_max)
        else:
            init_value=rand_init_norm
            if rand_init_norm is None:
                init_value=noise_norm
            noise_init=get_noise_init(norm_type, noise_norm, init_value, Xout)
            Xn = torch.clamp(Xout + noise_init, clip_X_min, clip_X_max)
    else:
        Xn = Xout.detach()
    #-----------------
    noise_new=(Xn-Xout).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise_new], lr=step)
    #-----------------
    if Y is None and loss_model is None:
        Z = run_model(model, Xin.detach())
        Z = Z.detach()
    #-----------------
    for n in range(0, max_iter):
        Xn = Xn.detach()
        Xn.requires_grad = True
        if loss_model is None:
            Zn = run_model(model, Xn)
            if Y is not None:
                loss = loss_fn(Zn, Y)
            else:
                loss = loss_fn(Zn, Z)
        else:
            loss=loss_model(model, Xin, Y, Xn)
        #---------------------------
        #loss.backward() will update W.grad
        grad_n=torch.autograd.grad(loss, Xn)[0]
        grad_n=normalize_grad_(grad_n, norm_type)
        if use_optimizer == True:
            noise_new.grad=grad_n.detach() #grad descent to minimize loss
            optimizer.step()
        else:
            Xnew = Xn.detach() - step*grad_n.detach()
            noise_new = Xnew-Xout
        #---------------------
        noise_new.data.clamp_(clip_E_min, clip_E_max)
        #print('norm a ', torch.norm(noise_new, p=norm_type).item())
        clip_norm_(noise_new, norm_type, noise_norm)        
        #print('norm b', torch.norm(noise_new, p=norm_type).item())
        Xn = torch.clamp(Xout+noise_new, clip_X_min, clip_X_max)
        #print('norm c', torch.norm(Xn-Xout, p=norm_type).item())
        noise_new.data -= noise_new.data-(Xn-Xout).data
        Xn=Xn.detach()
        #---------------------------
        if n in lr_schedule.keys():
            lr=lr_schedule[n]
            update_lr(optimizer, lr)
    #---------------------------
    return Xn
#%%

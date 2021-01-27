# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Guc, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""

import torch
#from TPS2D import TPS2D
from TPS3D import TPS2D
#%%
class PolyToImage2D(torch.nn.Module):
    def __init__(self, poly, image, origin, swap_xy, eps=1e-7):
        #poly.shape: batch x N x 2, poly is shape template 
        #if poly[0,0,:] is (x,y) then set swap_xy to False
        #if poly[0,0,:] is (y,x) then set swap_xy to True
        #image.shape: batch x C x H x W,  H is y-axis, W is x-axis, image is image template 
        #origin: image[:,:,0,0] in poly space, e.g. [0.5, 0.5] in matlab and sk-image-poly2mask
        #assume spacing is [1,1]
        #swap_h_w: swap h and w axis in poly and x input to forward
        super().__init__()
        self.poly=poly if swap_xy==False else torch.flip(poly, dims=[2])
        self.image=image
        self.origin=origin
        self.swap_xy=swap_xy
        self.eps=eps
        H=image.shape[2]
        W=image.shape[3]
        h=torch.linspace(origin[0], H-1+origin[0], H, dtype=poly.dtype, device=poly.device)
        w=torch.linspace(origin[1], W-1+origin[1], W, dtype=poly.dtype, device=poly.device)        
        h, w=torch.meshgrid(h, w)
        h=h.view(h.shape[0], h.shape[1], 1)
        w=w.view(w.shape[0], w.shape[1], 1)
        self.grid = torch.cat([w,h], dim=2) # [w, h] is [x, y] for grid_sample
        #print(self.grid.shape)
        #grid.shape: H x W x 2

    def forward(self, x):
        #x.shape: batch x N x 2        
        if self.swap_xy==True:
            x=torch.flip(x, dims=[2])
        poly=self.poly
        if poly.shape[0]==1:
            poly=poly.expand(x.shape[0], poly.shape[1], poly.shape[2])        
        image=self.image
        if image.shape[0]==1:
            image=image.expand(x.shape[0], image.shape[1], image.shape[2], image.shape[3])
        H=self.grid.shape[0]
        W=self.grid.shape[1]
        grid=self.grid.view(-1, 2)
        grid=grid.expand(x.shape[0], grid.shape[0], 2)
        tps=TPS2D(x, poly, self.eps)
        grid=tps(grid)
        #in grid_sample, left-top pixel [-1 -1], right-bottom pixel [1, 1]    
        w=grid[:,:,0:1]
        h=grid[:,:,1:2]          
        h=-1+(h-self.origin[0])*2/(H-1)        
        w=-1+(w-self.origin[1])*2/(W-1)        
        grid = torch.cat([w,h], dim=2)
        grid = grid.view(grid.shape[0], H, W, 2)
        x_image=torch.nn.functional.grid_sample(image, grid, align_corners=False)
        #print(x.requires_grad, grid.requires_grad, image.requires_grad, x_image.requires_grad)
        return x_image
#%%
import numpy as np
from scipy import ndimage
def distance_map(mask, normalize=False, signed=False):
    #mask.shape: batch x C x H x W or batch x C x D x H x W
    device=mask.device
    dtype=mask.dtype
    mask=mask.detach().cpu().numpy()
    map=[]
    for b in range(0, mask.shape[0]):
        map_b=[]
        for c in range(0, mask.shape[1]):        
            if signed == True:
                map_c1=ndimage.distance_transform_edt(mask[b,c])
                map_c2=ndimage.distance_transform_edt(1-mask[b,c])
                map_c=map_c1-map_c2
                if normalize == True:
                    map_c/=map_c1.max()
            else:
                map_c=ndimage.distance_transform_edt(mask[b,c])
                if normalize == True:
                    map_c/=map_c.max()
            map_b.append(map_c.reshape(1,1,map_c.shape[0], map_c.shape[1]))
        map_b=np.concatenate(map_b, axis=1)
        map.append(map_b)
    map=np.concatenate(map, axis=0)
    map=torch.tensor(map, dtype=dtype, device=device)
    return map
#%%
    
    
    
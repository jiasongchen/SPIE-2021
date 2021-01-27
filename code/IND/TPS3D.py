# -*- coding: utf-8 -*-
"""

"""

import torch
#%%
class TPS3D(torch.nn.Module):
    def __init__(self, source, target, eps=1e-7):
        super().__init__()
        #transform: from source to target
        #source.shape: batch x N x 3
        #target.shape: batch x N x 3
        self.source=source
        self.target=target
        self.eps=eps
        self.update_parameter()
        
    def update_parameter(self):
        dtype=self.source.dtype
        device=self.source.device
        batch_size=self.source.shape[0]
        N=self.source.shape[1]
        K=torch.cdist(self.source, self.source, p=2)
        ones=torch.ones((batch_size, N, 1), dtype=dtype, device=device)
        P=torch.cat([ones, self.source], dim=2)
        Pt=P.permute(0, 2, 1)
        zero4x4 = torch.zeros((batch_size,4,4), dtype=dtype, device=device)
        La = torch.cat([K, P], dim=2) 
        Lb = torch.cat([Pt, zero4x4], dim=2)
        L = torch.cat([La, Lb], dim=1)
        zero4x3=torch.zeros((batch_size,4,3), dtype=dtype, device=device)
        B = torch.cat([self.target, zero4x3], dim=1)
        #detach L, backprop through pinverse is not stable
        L=L.detach()
        Linv=torch.pinverse(L.to(torch.double)).to(L.dtype)
        self.param = torch.bmm(Linv, B)        

    def forward(self, x):
        #x.shape:  batch x N x 3
        K = torch.cdist(x, self.source, p=2)     
        #----prevent nan grad caused by sqrt in cdist
        eps=self.eps
        if eps > 0:
            Keps=(K-eps)*(K<eps)
            K.data.clamp_(min=eps)
            K=K+Keps.detach() 
        #----
        ones=torch.ones((x.shape[0], x.shape[1], 1), dtype=x.dtype, device=x.device)
        P = torch.cat([ones, x], dim=2)
        L = torch.cat([K, P], dim=2)
        y = torch.bmm(L, self.param)
        return y
#%% this is using TPS3D
'''
class TPS2D_a(torch.nn.Module):
    def __init__(self, source, target, eps=1e-7):
        super().__init__()
        #source.shape: batch x N x 2
        #target.shape: batch x N x 2
        dtype=source.dtype
        device=source.device
        batch_size=source.shape[0]
        N=source.shape[1]
        zeros=torch.zeros((batch_size, N, 1), dtype=dtype, device=device)
        self.source=torch.cat([source, zeros], dim=2)
        self.target=torch.cat([target, zeros], dim=2)
        self.tps=TPS3D(self.source, self.target, eps)
    def forward(self, x):
        #x.shape: batch x N x 2
        zeros=torch.zeros((x.shape[0], x.shape[1], 1), dtype=x.dtype, device=x.device)
        x=torch.cat([x, zeros], dim=2)
        y=self.tps(x)
        y=y[:,:,0:2]
        return y
'''
#%% this is using TPS3D formula
class TPS2D(torch.nn.Module):
    def __init__(self, source, target, eps=1e-7):
        super().__init__()
        #transform: from source to target
        #source.shape: batch x N x 2
        #target.shape: batch x N x 2
        self.source=source
        self.target=target
        self.eps=eps
        self.update_parameter()
        
    def update_parameter(self):
        dtype=self.source.dtype
        device=self.source.device
        batch_size=self.source.shape[0]
        N=self.source.shape[1]
        K=torch.cdist(self.source, self.source, p=2)
        ones=torch.ones((batch_size, N, 1), dtype=dtype, device=device)
        P=torch.cat([ones, self.source], dim=2)
        Pt=P.permute(0, 2, 1)
        zero3x3 = torch.zeros((batch_size,3,3), dtype=dtype, device=device)
        La = torch.cat([K, P], dim=2) 
        Lb = torch.cat([Pt, zero3x3], dim=2)
        L = torch.cat([La, Lb], dim=1)
        zero3x2=torch.zeros((batch_size,3,2), dtype=dtype, device=device)
        B = torch.cat([self.target, zero3x2], dim=1)
        #detach L, backprop through pinverse is not stable
        L=L.detach()
        Linv=torch.pinverse(L.to(torch.double)).to(L.dtype)
        self.param = torch.bmm(Linv, B)        

    def forward(self, x):
        #x.shape:  batch x N x 2
        K = torch.cdist(x, self.source, p=2)     
        #----prevent nan grad caused by sqrt in cdist
        eps=self.eps
        if eps > 0:
            Keps=(K-eps)*(K<eps)
            K.data.clamp_(min=eps)
            K=K+Keps.detach() 
        #----
        ones=torch.ones((x.shape[0], x.shape[1], 1), dtype=x.dtype, device=x.device)
        P = torch.cat([ones, x], dim=2)
        L = torch.cat([K, P], dim=2)
        y = torch.bmm(L, self.param)
        return y
#%%
import matplotlib.pyplot as plt
if __name__ == '__main__':
    N=176
    source=torch.rand((1,N,2))
    target=torch.rand((1,N,2))
    tps=TPS2D(source, target)
    #
    x=source+0.0*torch.rand((1,N,2))
    x.requires_grad=True
    y=tps(x)
    y.sum().backward()
    print(torch.isnan(x.grad).sum())
    #%%
    x=source#+0.1
    x.requires_grad=True
    K = torch.cdist(x, source, p=2)  
    K.sum().backward()
    print(torch.isnan(x.grad).sum())
    #%%
    source=source.detach().cpu().numpy()
    target=target.detach().cpu().numpy()
    x=x.detach().cpu().numpy()
    y=y.detach().cpu().numpy()
    fig, ax= plt.subplots()
    ax.plot(source[0,:,0], source[0,:,1], 'r.')
    ax.plot(target[0,:,0], target[0,:,1], 'b.')
    ax.plot(x[0,:,0], x[0,:,1], 'm.')
    ax.plot(y[0,:,0], y[0,:,1], 'mo', fillstyle='none')
    
    
    
    
    
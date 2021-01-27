# -*- coding: utf-8 -*-
"""
Adversarial Robustness Study of Convolutional Neural Network for Lumbar Disk Shape Reconstruction from MR images 
(Jiasong Chen, Linchen Qian, Timur Urakov, Weiyong Guc, Liang Liang at University of Miami)
published at SPIE Medical Imaging: Image Processing, 2021

"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from Lumbar_Dataset import DiskSet
from Lumbar_Dataset import DiskSet_example
from sklearn.decomposition import PCA
from PolyToImage2D import PolyToImage2D
#%%
class PCA_Aug:
    def __init__(self, n_components, c_max, flag, train, path, c_list=None):        
        if train == True:
            file='aug_data_example_train.txt'
        else:
            file='aug_data_example_test.txt'
        Dataset = DiskSet_example(path, file)
        loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size = 64, shuffle = False)    
        S=[]
        X=[]
        for batch_idx, (Xi, Si) in enumerate(loader):
            X.append(Xi.numpy())
            S.append(Si.numpy())
        X=np.concatenate(X, axis=0)
        S=np.concatenate(S, axis=0)
        pca = PCA(n_components=n_components)
        pca.fit(S.reshape(S.shape[0], -1))
        self.pca=pca
        self.P=torch.tensor(pca.components_.reshape(1,n_components,-1), dtype=torch.float32)
        self.V=torch.tensor(np.sqrt(pca.explained_variance_).reshape(1,n_components,1), dtype=torch.float32)
        self.Smean=torch.tensor(np.mean(S, axis=0, keepdims=True), dtype=torch.float32)
        self.X=torch.tensor(X, dtype=torch.float32)
        self.S=torch.tensor(S, dtype=torch.float32)
        self.n_components=n_components
        self.c_max=c_max
        self.flag=flag
        border=[]
        for x in [1, 128]:
            for y in [1, 32, 64, 96, 128]:
                border.append([x, y])
                #pass
        for y in [1, 128]:
            for x in [32, 64, 96]:
                border.append([x, y])
                #pass
        self.border=torch.tensor(border, dtype=torch.float32).view(1,-1,2)-0.5
        self.rng1=torch.Generator()
        self.rng1.manual_seed(1)
        self.rng2=torch.Generator()
        self.rng2.manual_seed(2)
        self.c_list=None
        if c_list is not None:
            if c_list =='auto5':
                 self.c_list=torch.linspace(-c_max, c_max, 5)
            else:
                self.c_list=torch.tensor(c_list)

    def to(self, sth):
        #sth is device or dtype
        self.P=self.P.to(sth)
        self.V=self.V.to(sth)
        self.Smean=self.Smean.to(sth)
        self.border=self.border.to(sth)
        if self.c_list is not None:
            self.c_list=self.c_list.to(sth)
        if isinstance(sth, torch.device):
            self.rng1=torch.Generator(sth)
            self.rng1.manual_seed(1)
            self.rng2=torch.Generator(sth)
            self.rng2.manual_seed(2)

    def add_boarder(self, s):        
        border= self.border.expand(s.shape[0], self.border.shape[1], self.border.shape[2])
        s=torch.cat([s, border], dim=1)
        return s

    def generate_shape(self, batch_size):
        if self.c_list is not None:
            idx=torch.randint(0, self.c_list.shape[0], (batch_size, self.n_components, 1), 
                              device=self.P.device, generator=self.rng2)
            c=self.c_list[idx.view(-1)].view(idx.shape)
        else:
            c=-self.c_max+2*self.c_max*torch.rand(batch_size, self.n_components, 1, 
                                                  dtype=self.P.dtype, device=self.P.device, generator=self.rng2)
        s=torch.sum(c*self.V*self.P, dim=1)
        s=s.view(batch_size,-1,2)+self.Smean
        return s

    def generate_image(self, s, x_real, s_real):
        poly2image=PolyToImage2D(self.add_boarder(s_real), x_real, origin=[0.5, 0.5], swap_xy=False)
        x=poly2image(self.add_boarder(s))
        return x

    def make_batch(self, batch_size):        
        dtype=self.P.dtype
        device=self.P.device
        if self.flag==0:
            idx=torch.randint(0, self.X.shape[0], (batch_size,), device=device, generator=self.rng1)
            x_real=self.X[idx].to(dtype).to(device)
            s_real=self.S[idx].to(dtype).to(device)
            s=self.generate_shape(batch_size)
            x=self.generate_image(s, x_real, s_real)
        elif self.flag==1:
            idx=torch.randint(0, self.X.shape[0], (1,), device=device, generator=self.rng1)
            x_real=self.X[idx].to(dtype).to(device)
            s_real=self.S[idx].to(dtype).to(device)
            s=self.generate_shape(batch_size)
            x=self.generate_image(s, x_real, s_real)
        elif self.flag==2:
            idx=torch.randint(0, self.X.shape[0], (batch_size,), device=device, generator=self.rng1)
            x_real=self.X[idx].to(dtype).to(device)
            s_real=self.S[idx].to(dtype).to(device)
            s=self.generate_shape(1)
            s=s.expand(batch_size, -1, 2)
            x=self.generate_image(s, x_real, s_real)
        else:
            raise ValueError('uknown flag')
        s=s[:,0:176,:]
        return x, s
    
    def __call__(self, batch_size):
        return self.make_batch(batch_size)
#%%
class PCA_Aug_Dataset(torch.utils.data.Dataset):
    def __init__(self, n_sets, n_samples_per_set, device, filename,
                 n_components, c_max, flag, train, path, c_list=None, return_idx=False):
        super().__init__()
        self.pca_aug=PCA_Aug(n_components, c_max, flag, train, path, c_list)
        self.pca_aug.to(device)
        self.current_set_idx=-1
        self.n_sets=n_sets
        self.n_samples_per_set=n_samples_per_set
        self.filename=filename
        self.buffer=None
        self.rng=np.random.RandomState(0)
        self.pin_memory=False
        if torch.cuda.is_available():
            self.pin_memory=True
        self.return_idx=return_idx

    def generate_one_set(self):
        data_x=[]
        data_s=[]
        N=self.n_samples_per_set//64
        for n in range(0, N):
            x, s = self.pca_aug(64)
            x, s = x.cpu(), s.cpu()
            data_x.append(x)
            data_s.append(s)
        data_x=torch.cat(data_x, dim=0)
        data_s=torch.cat(data_s, dim=0)
        data = [data_x, data_s]
        return data

    def generate(self, set_idx_start, set_idx_end):
        for idx in range(set_idx_start, set_idx_end):
            t0=time.time()
            data=self.generate_one_set()
            filename=self.filename+'_set'+str(idx)+'.pt'
            torch.save(data, filename)
            duration=time.time()-t0
            print('generate data in PCA_Aug_Dataset', filename, 'time cost', duration)

    def load_one_set(self, set_idx):
        if self.filename is None:
            t0=time.time()
            data=self.generate_one_set()
            duration=time.time()-t0
            print('generate data in PCA_Aug_Dataset, None, time cost', duration)
        else:
            filename=self.filename+'_set'+str(set_idx)+'.pt'
            try:
                data=torch.load(filename)
            except:
                t0=time.time()
                data=self.generate_one_set()
                duration=time.time()-t0
                print('generate data in PCA_Aug_Dataset', filename, 'time cost', duration)
                try:
                    torch.save(data, filename)
                except:
                    print('PCA_Aug_Dataset can not save', filename)
        return data

    def get_set_idx_by_sample_idx(self, sample_idx):
        return sample_idx//self.n_samples_per_set

    def __len__(self):
        return self.n_sets*self.n_samples_per_set

    def __getitem__(self, idx):
        #idx is sample_idx
        set_idx=self.get_set_idx_by_sample_idx(idx)
        #because of random shuffle, it is not good to directly use PCA_Aug_Dataset
        if set_idx != self.current_set_idx:
            self.buffer=self.load_one_set(set_idx)
            self.buffer.append(torch.arange(0, len(self.buffer[0])))
            self.current_set_idx=set_idx
        x=self.buffer[0][idx]
        s=self.buffer[1][idx]
        if self.return_idx == False:
            return x, s
        else:
            return x, s, idx
#%%
class PCA_Aug_Dataloader:
    def __init__(self, n_epochs, n_batches, batch_size, device, shuffle, filename, return_sample_idx=False,
                 n_components=None, c_max=None, flag=None, train=None, path=None, c_list=None):
        super().__init__()
        self.pca_aug=None
        if n_components is not None:
            self.pca_aug=PCA_Aug(n_components, c_max, flag, train, path, c_list)
            self.pca_aug.to(device)
        self.n_epochs=n_epochs
        self.n_batches=n_batches
        self.batch_size=batch_size
        self.filename=filename
        self.shuffle=shuffle
        self.buffer=None
        self.epoch=-1
        self.epoch_list=np.arange(0, n_epochs)
        self.rng=np.random.RandomState(0)
        self.pin_memory=False
        if torch.cuda.is_available():
            self.pin_memory=True
        self.return_sample_idx=return_sample_idx

    def __len__(self):
        return self.n_batches

    def generate_one_epoch(self):
        data_x=[]
        data_s=[]
        for n in range(0, self.n_batches):
            x, s = self.pca_aug(self.batch_size)
            x, s = x.cpu(), s.cpu()
            data_x.append(x)
            data_s.append(s)
        data_x=torch.cat(data_x, dim=0)
        data_s=torch.cat(data_s, dim=0)
        data = [data_x, data_s]
        return data

    def generate(self, epoch_start, epoch_end):
        for epoch in range(epoch_start, epoch_end):
            t0=time.time()
            data=self.generate_one_epoch()
            filename=self.filename+'_epoch'+str(epoch)+'.pt'
            torch.save(data, filename)
            duration=time.time()-t0
            print('generate data in PCA_Aug_Dataloader', filename, 'time cost', duration)
 
    def load_one_epoch(self, epoch):
        if self.filename is None:
            t0=time.time()
            data=self.generate_one_epoch()
            duration=time.time()-t0
            print('generate data in PCA_Aug_Dataloader, None, time cost', duration)
        else:
            filename=self.filename+'_epoch'+str(epoch)+'.pt'
            try:
                data=torch.load(filename)
            except:
                t0=time.time()
                data=self.generate_one_epoch()
                duration=time.time()-t0
                print('generate data in PCA_Aug_Dataloader', filename, 'time cost', duration)
                try:
                    torch.save(data, filename)
                except:
                    print('PCA_Aug_Dataloader can not save', filename)
        return data

    def get_n_samples(self):
        return self.n_epochs*self.n_batches*self.batch_size

    def __getitem__(self, idx):
        if idx == 0:
            self.epoch+=1
            if self.epoch >= self.n_epochs:
                self.epoch=0
                if self.shuffle == True:
                    self.rng.shuffle(self.epoch_list)
            self.buffer=self.load_one_epoch(self.epoch_list[self.epoch])
            self.buffer.append(torch.arange(0, len(self.buffer[0])))
            if self.shuffle == True:
                idxlist=np.arange(0, len(self.buffer[0]))
                self.rng.shuffle(idxlist)
                self.buffer[0]=self.buffer[0][idxlist]
                self.buffer[1]=self.buffer[1][idxlist]
                self.buffer[2]=self.buffer[2][idxlist]
        elif idx >= self.n_batches:
            raise IndexError
        x=self.buffer[0][idx*self.batch_size:(idx+1)*self.batch_size].contiguous()
        s=self.buffer[1][idx*self.batch_size:(idx+1)*self.batch_size].contiguous()
        sample_idx=self.epoch*self.n_batches*self.batch_size+self.buffer[2][idx*self.batch_size:(idx+1)*self.batch_size]
        if self.pin_memory == True:
            x=x.pin_memory()
            s=s.pin_memory()
            sample_idx=sample_idx.pin_memory()
        if self.return_sample_idx == False:
            return x, s
        else:
            return x, s, sample_idx
#%%
if __name__ == '__main__':
    # loader_train=PCA_Aug_Dataloader(n_epochs=100, n_batches=100, batch_size=64, device=torch.device("cuda:1"), shuffle=True,
    #                             filename='../data/'+'pca_aug_P',
    #                             n_components=30, c_max=2, flag=0, train=True, path='../data/')
    from scipy.io import savemat,loadmat
    import matplotlib.pyplot as plt
    # data = torch.load(r'C:\Users\Liang\Desktop\code\code\data\pca_aug_P_epoch0.pt')
    # img=data[0].numpy()
    # shape = data[1].numpy()
    # for i in range(0,len(img)):
    #     fig, ax = plt.subplots()
    #     x=np.squeeze(img[i])
    #     s=np.squeeze(shape[i])
    #     ax.plot(s[:,0],s[:,1])
    #     ax.imshow(x,cmap='gray')
    #     plt.show()
        
    #     dat ={'img':x,'shape':s}

    #     savemat('data/%s.mat'%(i),dat)
    import os
    d = r"C:\Users\Liang\Desktop\code\code\IND\data"
    file =open("aug_data_example.txt",'w+')

    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        file.write(full_path + '\n')
        
        if os.path.isfile(full_path):
            print (full_path)
    file.close()
    
    # fig, ax = plt.subplots()
    # pp=loadmat('abc.mat')
    # x=pp['img']
    # s=pp['shape']    
    # ax.plot(s[:,0],s[:,1])
    # ax.imshow(x,cmap='gray')
    # plt.show()
        
        
        
        
        
        
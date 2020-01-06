import os
from torch.utils.data.dataset import Dataset
from torch.utils import data
import torch.cuda
import h5py
import numpy as np
import random
from nilearn import image
import pandas as pd




class fMRIDataset_CRNN(Dataset):
    def __init__(self, datadir, s_sz, ID, T, csv,rep):
        self.datadir = datadir  # downsampled data folder dir
        self.s_sz = s_sz  # num of channels
        self.ID = ID  # ID array of subjecst
        self.T = T
        self.csv = csv
        self.rep = rep

    def __getitem__(self, index):
        id = self.ID[index]
        filename = 'NYU_00'+str(id)+'_func_preproc.h5'
        hf = h5py.File(os.path.join(self.datadir, filename),'r')
        fMRI = hf['fMRI'][:]
        fMRI = np.moveaxis(fMRI,-1,0)
        #mask  = fMRI<=0
        # x = fMRI[fMRI > 0]
        # x_std = np.std(x)
        # x_mean = np.mean(x)
        #fMRI = (fMRI - 5390.37)/6082.68
        #fMRI = np.arctan(fMRI)
        #fMRI[mask] = -1000
        fMRI = np.nan_to_num(fMRI)
        fMRI = fMRI.astype('float32')
        #fMRI -= np.mean(fMRI)
        fMRI /= np.max(abs(fMRI))
        #fMRI = (fMRI - np.min(fMRI)) / np.ptp(fMRI) #[0,1]
        data_blanket = []
        for i in range(self.rep):
            ind = random.sample(range(0, self.T - self.s_sz + 1), 1)[0]  # sample index
            data_blanket.append(fMRI[ind:ind + self.s_sz, :])
        onebatch_x = np.stack(data_blanket)
        onebatch_y = np.repeat(self.csv[self.csv['SUB_ID']==id]['DX_GROUP'].values[0]%2, self.rep)
        return onebatch_x, onebatch_y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ID)


def get_loader(datadir, win_size, ID, T):
    # para setting
    use_cuda = torch.cuda.is_available()
    print('use gpu:',use_cuda)
    kwargs = {'num_workers': 24, 'pin_memory': True} if use_cuda else {}  ##num_workers

    # define dataloader
    data_set = fMRIDataset_CRNN(datadir, win_size, ID, T)
    generator = data.DataLoader(dataset=data_set, batch_size=10, shuffle=True, **kwargs)

    return generator


############################# Dataloder Testing Script##################
'''
from scipy.io import loadmat
# define directory
MAT_dir = '../data/UM/MAT/'
data_dir = '../data/UM/UM/'

# load subjects ID
IDmat = loadmat(os.path.join(MAT_dir, 'subjects.mat'))
ID = IDmat['subjects']

# data loading parameters
win_size = 5  # num of channel input
T = 293  # total length of fMRI
num_rep = T // win_size  # num of repeat the ID
ID_rep = np.repeat(ID,
                   3 * num_rep)  # repeat the ID, in order to visit all the volumes in fMRI, this will be input to the dataloader
max_epoch = 10

for epoch in range(max_epoch):
    i = 0
    generator = get_loader(data_dir, win_size, ID_rep, T)
    for x, y in generator:
        print(i, x.size(), y)
'''








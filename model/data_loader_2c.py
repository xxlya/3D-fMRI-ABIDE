import os
from torch.utils.data.dataset import Dataset
from torch.utils import data
import torch.cuda
import h5py
import numpy as np
import random
from nilearn import image
import pandas as pd
from opts import parse_opts
import deepdish as dd

opt = parse_opts()


class fMRIDataset_2C(Dataset):
    def __init__(self, datadir, win_size, ID, T, csv):
        self.datadir = datadir  # downsampled data folder dir
        self.nch = win_size  # num of channels
        self.ID = ID  # ID array of subjecst
        self.T = T
        self.csv = csv

    def __getitem__(self, index):
        id = self.ID[index]
        filename = str(id)+'.h5'
        hf = dd.io.load(os.path.join(self.datadir, filename))
        avg = hf['avg']
        std = hf['std']
        y = hf['label']
        avg /= np.max(abs(avg))
        std /= np.max(std)
        x = np.stack((avg,std))

        return x,y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ID)


def get_loader(datadir, win_size, ID, T, csv):
    # para setting
    use_cuda = torch.cuda.is_available()
    print('use gpu:',use_cuda)
    kwargs = {'num_workers': 24, 'pin_memory': True} if use_cuda else {}  ##num_workers

    # define dataloader
    data_set = fMRIDataset_2C(datadir, win_size, ID, T, csv)
    generator = data.DataLoader(dataset=data_set, batch_size=10, shuffle=True, **kwargs)

    return generator


############################# Dataloder Testing Script##################
'''
csv = pd.read_csv(opt.csv_dir)

from scipy.io import loadmat
# define directory
data_dir = opt.datadir

# load subjects ID
ID = 50953

# data loading parameters
win_size = 2  # num of channel input
T = 175  # total length of fMRI
num_rep = T // win_size  # num of repeat the ID
ID_rep = np.repeat(ID,
                   3 * num_rep)  # repeat the ID, in order to visit all the volumes in fMRI, this will be input to the dataloader
max_epoch = 10

for epoch in range(max_epoch):
    i = 0
    generator = get_loader(data_dir, win_size, ID_rep, T, csv)
    for x, y in generator:
        print(i, x.size(), y)
'''








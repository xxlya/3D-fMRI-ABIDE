import csv
from sklearn.model_selection import StratifiedKFold
import numpy as np
import deepdish as dd
import os
from torch import tensor
import h5py
import random
from nilearn import image
import pandas as pd
from scipy.io import loadmat

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    #n_correct_elems = correct.float().sum().data[0]

    return float(n_correct_elems) / batch_size


def data_split(total,pos,neg,fold,savedir):
    '''

    :param total: total number of subjects
    :param pos: total number of positive subject
    :param neg: total number of negative subject
    :param fold: number of folds
    :param savedir: save directory
    :return:
    '''
    x_ind = range(0,total)
    y_ind = np.concatenate((np.ones(pos),np.zeros(neg))) #patient control
    kfold = fold
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=7)
    skf2 = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=3)
    test_index = list()
    train_index = list()
    val_index = list()
    i = 0
    for a, b in skf.split(x_ind, y_ind):
        test_index.append(b)
        temp1, temp2 = list(skf2.split(a, y_ind[a]))[i]
        c = a[temp1]
        d = a[temp2]
        train_index.append(c)
        val_index.append(d)
        i = i + 1
    dd.io.save(os.path.join(savedir,'train_index.h5'),{'id':train_index})
    dd.io.save(os.path.join(savedir,'test_index.h5'),{'id':test_index})
    dd.io.save(os.path.join(savedir,'val_index.h5'),{'id':val_index})

def get_NYU_test_data(datadir, ID, T, nch, csv):
    '''

    :param datadir: testing data directory
    :param ID: filename of the subject
    :param T: total fMRI length
    :param nch: number of subsequntial frames used as input
    :return:
    '''
    filename = 'NYU_00' + str(ID) + '_func_preproc.nii.gz'
    hf = image.smooth_img(os.path.join(datadir, filename), fwhm=3)
    fMRI = hf._data
    fMRI = np.moveaxis(fMRI, -1, 0)
    fMRI = np.nan_to_num(fMRI)
    fMRI = fMRI.astype('float32')
    fMRI -= np.mean(fMRI)
    # fMRI /= np.max(abs(fMRI))
    # fMRI = (fMRI - np.min(fMRI)) / np.ptp(fMRI) #[0,1]
    ind = random.sample(range(0, T - nch + 1), 1)[0]  # sample index
    onebatch_x = fMRI[ind:ind + nch, :]
    onebatch_y = csv[csv['SUB_ID'] == ID]['DX_GROUP'].values[0] % 2
    return onebatch_x, onebatch_y

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1

    return TP, FP, TN, FN


from opts import parse_opts
opt = parse_opts()

def get_train_mean_std(fold):
    train_index = dd.io.load(os.path.join(opt.MAT_dir, 'test_index.h5'))['id'][fold]
    csv0 =  pd.read_csv(opt.csv_dir)
    ID = csv0['SUB_ID'].values
    mean_list = []
    std_list = []
    for i in train_index:
        filename = 'NYU_00' + str(ID[i]) + '_func_preproc.h5'
        hf = h5py.File(os.path.join(opt.datadir, filename),'r')
        fMRI = hf['fMRI'][:]
        fMRI = np.moveaxis(fMRI,-1,0)
        x = fMRI[fMRI > 0]
        mean_list.append(np.std(x))
        std_list.append(np.mean(x))
    mean_arr = sum(mean_list)/len(mean_list)
    std_arr = sum(std_list)/len(std_list)
    return sum(mean_list)/len(mean_list),sum(std_list)/len(std_list)

def train_test_split(nfold):
    mat = loadmat('/basket/Biopoint_3DConv_Classification_pytorch/MAT/subject.mat')
    con = mat['con'].squeeze()
    pat = mat['pat'].squeeze()
    X = np.concatenate((con,pat))
    y = np.concatenate((np.ones_like(con),np.ones_like(pat)))
    skf = StratifiedKFold(n_splits=nfold, random_state = 42, shuffle= True)
    fold = dict()
    for i,(train_index, test_index) in enumerate(skf.split(X, y)):
        fold[i] = dict()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold[i]['X_train'] = X_train
        fold[i]['X_test'] = X_test
    dd.io.save('MAT/fold_split.h5',fold)

def train_test_split_rep(nfold,rep):
    ''' data is augmented'''
    csv_dir = '/basket/ABIDE_Data/data/RAW/NYU/MAT/NYU.csv'
    csv = pd.read_csv(csv_dir)
    con = csv[csv['DX_GROUP']==2]['SUB_ID'].values
    pat = csv[csv['DX_GROUP']==1]['SUB_ID'].values
    X = np.concatenate((con,pat))
    y = np.concatenate((np.ones_like(con),np.ones_like(pat)))
    skf = StratifiedKFold(n_splits=nfold, random_state = 42, shuffle= True)
    fold = dict()
    for i,(train_index, test_index) in enumerate(skf.split(X, y)):
        fold[i] = dict()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_rep = []
        X_test_rep = []
        for sub in list(X_train):
            X_train_rep.append([str(sub) + '_' + str(r) for r in range(rep)])
        for sub in list(X_test):
            X_test_rep.append([str(sub) + '_' + str(r) for r in range(rep)])
        X_train_rep = np.concatenate(X_train_rep)
        X_test_rep = np.concatenate(X_test_rep)
        fold[i]['X_train'] = X_train_rep
        fold[i]['X_test'] = X_test_rep
    dd.io.save('MAT/fold_split_rep.h5',fold)
# m,s = get_train_mean_std(7)
# print(m,s)
# # 5390.370947202441 6082.680453672701

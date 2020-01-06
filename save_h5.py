import deepdish as dd
import multiprocessing
from nilearn import image
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import h5py

csv_dir = '/basket/ABIDE_Data/data/RAW/NYU/MAT/NYU.csv'
csv = pd.read_csv(csv_dir)
data_dir = '/basket/ABIDE_Data/data/RAW/NYU/'
save_dir = '/basket/ABIDE_Data/data/RAW/NYU/h5_2c_win5'

def save_h5(sub_id):
    filename = 'NYU_00' + str(sub_id) + '_func_preproc.nii.gz'
    h5_name = 'NYU_00' + str(sub_id) + '_func_preproc.h5'
    hf = image.smooth_img(os.path.join(data_dir, filename), fwhm=3)
    fMRI = hf._data
    dd.io.save(os.path.join(save_dir,h5_name),{'fMRI':fMRI})

cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)

def save_h5_down(sub_id):
    filename = 'NYU_00' + str(sub_id) + '_func_preproc.nii.gz'
    h5_name = 'NYU_00' + str(sub_id) + '_func_preproc.h5'
    hf = image.smooth_img(os.path.join(data_dir, filename), fwhm=3)
    fMRI = hf._data
    shape = [61,73,61]
    steps = [0.5, 0.5, 0.5]  # original step sizes
    x, y, z = [steps[k] * np.arange(shape[k]) for k in range(3)]  # original grid
    new_fMRI = np.zeros((30,36,30,175))
    for t in range(175):
        f = RegularGridInterpolator((x, y, z), fMRI[:,:,:,t])  # interpolator
        dx, dy, dz = 1.0, 1.0, 1.0  # new step sizes
        new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]  # new grid
        new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
        new_values = f(new_grid)
        new_fMRI[:,:,:,t] = new_values
    dd.io.save(os.path.join(save_dir,h5_name),{'fMRI':new_fMRI})

def moving(a, n):
    res_std = np.zeros((a.shape[0],a.shape[1],a.shape[2],a.shape[3]-n+1))
    res_avg = np.zeros((a.shape[0],a.shape[1],a.shape[2],a.shape[3]-n+1))
    for i in range(0,a.shape[-1]-n+1):
        res_avg[:,:,:,i] = np.mean(a[:,:,:,i:i+n],axis=-1)
        res_std[:,:,:,i] = np.std(a[:,:,:,i:i+n],axis=-1)
    return res_avg, res_std

def save_h5_2c(sub_id):
    filename = 'NYU_00' + str(sub_id) + '_func_preproc.h5'
    h5_name = 'NYU_00' + str(sub_id) + '_func_preproc.h5'
    hf = h5py.File(os.path.join(data_dir, 'h5_downsample', filename), 'r')
    #hf = dd.io.load(os.path.join(data_dir, 'h5_downsample',filename))
    res_avg, res_std = moving(hf['fMRI'].value,n=5)
    res_std = np.nan_to_num(res_std)
    res_avg = np.nan_to_num(res_avg)
    hf.close()
    label = csv[csv['SUB_ID']==sub_id]['DX_GROUP'].values[0]%2
    for i in range(res_avg.shape[-1]):
        dd.io.save(os.path.join(save_dir,'NYU_00' + str(sub_id)+'_'+str(i)+'.h5'),{'avg':res_avg[:,:,:,i], 'std':res_std[:,:,:,i], 'label': label, 'id': sub_id})



cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)

#save_h5_2c(csv['SUB_ID'].values[0])


import timeit

start = timeit.default_timer()

pool.map(save_h5_2c,list(csv['SUB_ID'].values))

stop = timeit.default_timer()

print('Time: ', stop - start)
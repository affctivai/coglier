import os
from os.path import join
import pandas as pd
import numpy as np
from scipy import io
import re
from torcheeg.datasets.module import SEEDDataset

#  neutral, sad, fear, and happy
# segmentation x: (samples, 14, segment size), y: (samples, 2)  ## [label, subID]
## if window: 256, stride: 128 -> x: (15718, 62, 256), y: (15718, 2)
def save_datas_seg(window, stride, data_dir, saved_dir):
    print('Segmentation x: (samples, 62, segment size), y: (samples, 2)')

    labels = [-1, 1, 0, 2, 2, 0, 1, 2, 0, 1, 1, 0, 2, 0, 1, 2]

    dir_list = os.listdir(data_dir)
    skip_set = ['label.mat', 'readme.txt']
    dir_list = [f for f in dir_list if f not in skip_set]

    sub_dir_list = [[] for _ in range(0,16)]

    for dir_name in dir_list:
        sub_num = int(dir_name.split('_')[0])
        sub_dir_list[sub_num].append(dir_name)

    sub_list = [i for i in range(1,16)]
    
    for subidx in sub_list:
        x, y = [], []

        for session in range(0,3):
            path = join(data_dir, sub_dir_list[subidx][session])
            datas = io.loadmat(path)
            
            trial_name_ids = [(trial_name, int(re.findall(r".*_eeg(\d+)", trial_name)[0]))
                for trial_name in datas.keys() if 'eeg' in trial_name]
            
            for trial_name, trial_id in trial_name_ids:
                idx = 0
                data = datas[trial_name]
                time_size = len(data[0])
                while idx + window < time_size:
                    seg = data[:, idx : idx+window]
                    x.append(seg)
                    y.append([labels[trial_id], subidx]) # 데이터마다 subID
                    idx += stride
        x = np.array(x)
        x = np.array(x, dtype='float16')
        np.nan_to_num(x, copy=False)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subidx).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')

from utils.transform import BandDifferentialEntropy
# segmentation -> DE(BandDifferentialEntropy)
# x: (samples, 62, segment size) -> (samples, 62, 4(frequency))   y: (samples, 2) ## [label, subID]
## if window: 256, stride: 128 -> x: (15718, 62, 4), y: (15718, 2)
def save_datas_seg_DE(window, stride, data_dir, saved_dir):
    print('Segmentation with DE x: (samples, 62, 4), y: (samples, 2)')

    bde = BandDifferentialEntropy()

    labels = [-1, 1, 0, 2, 2, 0, 1, 2, 0, 1, 1, 0, 2, 0, 1, 2]

    dir_list = os.listdir(data_dir)
    skip_set = ['label.mat', 'readme.txt']
    dir_list = [f for f in dir_list if f not in skip_set]

    sub_dir_list = [[] for _ in range(0,16)]

    for dir_name in dir_list:
        sub_num = int(dir_name.split('_')[0])
        sub_dir_list[sub_num].append(dir_name)

    # sub_list = [i for i in range(1,16)]
    sub_list = [14, 15]

    for subidx in sub_list:
        x, y = [], []

        for session in range(0,3):
            path = join(data_dir, sub_dir_list[subidx][session])
            datas = io.loadmat(path)
            
            trial_name_ids = [(trial_name, int(re.findall(r".*_eeg(\d+)", trial_name)[0]))
                for trial_name in datas.keys() if 'eeg' in trial_name]
            
            for trial_name, trial_id in trial_name_ids:
                idx = 0
                data = datas[trial_name]
                time_size = len(data[0])
                while idx + window < time_size:
                    seg = data[:, idx : idx+window]
                    x.append(bde.apply(seg))
                    y.append([labels[trial_id], subidx]) # 데이터마다 subID
                    idx += stride
        x = np.array(x)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subidx).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')


# -----------------------------------------main---------------------------------------------------
# source data folder location
DATAS = join(os.getcwd(),'datasets',"SEED")
DATA = os.path.join(os.getcwd(),"..","..", "..", "..", "dataset", "SEED", "Preprocessed_EEG")

WINDOW = 128 * 2
STRIDE = 128

# -----------------------------------------save data-------------------------------------------------
# path to save preprocessed data(.npz format)
saved_dir = join(DATAS, 'npz', 'Preprocessed')

# There are 2 methods
save_datas_seg(WINDOW, STRIDE, DATA,join(saved_dir, 'seg'))
## DE calculation takes a time. be careful
# save_datas_seg_DE(WINDOW, STRIDE, DATA, join(saved_dir, 'seg_DE'))

# -----------------------------------------check---------------------------------------------------
# Save the bar graph of the number of labels per class
from utils.tools import getFromnpz, plot_VA

# load data
# saved_dir = join(DATAS, 'npz', 'seg_DE')

# sublists = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# datas_v, targets_v = getFromnpz(saved_dir, sublists, out=False, cla='v')
# datas_a, targets_a = getFromnpz(saved_dir, sublists, out=False, cla='a')
# vals, count_v = np.unique(targets_v[:, 0], return_counts=True)
# aros, count_a = np.unique(targets_a[:, 0], return_counts=True)

# subIDs, countss_v = np.unique(targets_v[:, 1], return_counts=True)
# subIDs, countss_a = np.unique(targets_a[:, 1], return_counts=True)

# print(f'data_v shape: {datas_v.shape} target_v shape: {targets_v.shape}')
# print(f'data_a shape: {datas_a.shape} target_a shape: {targets_a.shape}')
# print(f'valence {vals} \t {count_v}')
# print(f'arousal {aros} \t {count_a}')
# print(f'Num of data per subject {subIDs} \t {countss_v}') # subIDs

# plot_VA(vals, count_v, aros, count_a, path=join(DATAS,'GAMEEMO_npz','seg_DE.png'))
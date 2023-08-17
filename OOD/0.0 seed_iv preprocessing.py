import os
from os.path import join
import pandas as pd
import numpy as np
from scipy import io
import re
import argparse

#  neutral, sad, fear, and happy
session_label = [   
    [-1,1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    [-1,2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    [-1,1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
]

# No segmentation x: (4, 62, *), y: (4,)
def save_datas_noseg(data_dir, saved_dir):
    print('Segmentation x: (samples, 62, segment size), y: (samples, 2)')

    dir_list = []
    for i in range(1,4):
        path = join(data_dir, str(i))
        tmp = os.listdir(path)
        dir_list.append(tmp)
    subnums = []
    for data in dir_list[0]:
        subnums.append(int(data.split('_')[0]))

    for subidx in range(0,15):
        print('sub ID:',subnums[subidx], end=' ')
        x, y = [], []

        for session in range(1,4):
            path = join(data_dir, str(session), dir_list[session-1][subidx])
            datas = io.loadmat(path)
            trial_name_ids = [(trial_name, int(re.findall(r".*_eeg(\d+)", trial_name)[0]))
                for trial_name in datas.keys() if 'eeg' in trial_name]
            for trial_name, trial_id in trial_name_ids:
                data = datas[trial_name]
                time_size = len(data[0])
                seg = data[:, :]
                print(seg.shape)
                
                x.append(seg)
                y.append([session_label[session-1][trial_id], subnums[subidx]]) # 데이터마다 subID
        x = np.array(x, dtype='float16')
        np.nan_to_num(x, copy=False)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subnums[subidx]).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')

# segmentation x: (samples, 14, segment size), y: (samples, 2)  ## [label, subID]
## if window: 256, stride: 128 -> x: (15718, 62, 256), y: (15718, 2)
def save_datas_seg(window, stride, data_dir, saved_dir):
    print('Segmentation x: (samples, 62, segment size), y: (samples, 2)')

    dir_list = []
    for i in range(1,4):
        path = join(data_dir, str(i))
        tmp = os.listdir(path)
        dir_list.append(tmp)
    subnums = []
    for data in dir_list[0]:
        subnums.append(int(data.split('_')[0]))

    for subidx in range(0,15):
        print('sub ID:',subnums[subidx], end=' ')
        x, y = [], []

        for session in range(1,4):
            path = join(data_dir, str(session), dir_list[session-1][subidx])
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
                    y.append([session_label[session-1][trial_id], subnums[subidx]]) # 데이터마다 subID
                    idx += stride
        x = np.array(x, dtype='float16')
        np.nan_to_num(x, copy=False)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subnums[subidx]).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')

from utils.transform import BandDifferentialEntropy
# segmentation -> DE(BandDifferentialEntropy)
# x: (samples, 62, segment size) -> (samples, 62, 4(frequency))   y: (samples, 2) ## [label, subID]
## if window: 256, stride: 128 -> x: (15718, 62, 4), y: (15718, 2)
def save_datas_seg_DE(window, stride, data_dir, saved_dir):
    print('Segmentation with DE x: (samples, 62, 4), y: (samples, 2)')

    bde = BandDifferentialEntropy()
    dir_list = []
    for i in range(1,4):
        path = join(data_dir, str(i))
        tmp = os.listdir(path)
        dir_list.append(tmp)
    subnums = []
    for data in dir_list[0]:
        subnums.append(int(data.split('_')[0]))

    sublists = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

    # for subidx in range(0,15):
    for subidx in sublists:
        print('sub ID:',subnums[subidx], end=' ')
        x, y = [], []

        for session in range(1,4):
            path = join(data_dir, str(session), dir_list[session-1][subidx])
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
                    y.append([session_label[session-1][trial_id], subnums[subidx]]) # 데이터마다 subID
                    idx += stride
        x = np.array(x)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subnums[subidx]).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')


from utils.transform import BandPowerSpectralDensity
def save_datas_seg_PSD(window, stride, data_dir, saved_dir):
    print('Segmentation with PSD x: (samples, 62, 4), y: (samples, 2)')

    psd = BandPowerSpectralDensity()
    dir_list = []
    for i in range(1,4):
        path = join(data_dir, str(i))
        tmp = os.listdir(path)
        dir_list.append(tmp)
    subnums = []
    for data in dir_list[0]:
        subnums.append(int(data.split('_')[0]))

    sublists = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

    # for subidx in range(0,15):
    for subidx in sublists:
        print('sub ID:',subnums[subidx], end=' ')
        x, y = [], []

        for session in range(1,4):
            path = join(data_dir, str(session), dir_list[session-1][subidx])
            datas = io.loadmat(path)
            trial_name_ids = [(trial_name, int(re.findall(r".*_eeg(\d+)", trial_name)[0]))
                for trial_name in datas.keys() if 'eeg' in trial_name]
            for trial_name, trial_id in trial_name_ids:
                idx = 0
                data = datas[trial_name]
                time_size = len(data[0])
                while idx + window < time_size:
                    seg = data[:, idx : idx+window]
                    x.append(psd.apply(seg))
                    y.append([session_label[session-1][trial_id], subnums[subidx]]) # 데이터마다 subID
                    idx += stride
        x = np.array(x)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subnums[subidx]).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')
# -----------------------------------------save data-------------------------------------------------
# path to save preprocessed data(.npz format)
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, default="/mnt/data") # source data folder location
parser.add_argument("--window", type=int, default=400)
parser.add_argument("--stride", type=int, default=200)
parser.add_argument("--method", type=str, default="seg", help='noseg, seg, PSD, DE')
args = parser.parse_args()

SRC = args.src_dir
WINDOW = args.window
STRIDE = args.stride
METHOD = args.method
src_dir = join(SRC, 'SEED_IV', 'eeg_raw_data')
saved_dir = join(os.getcwd(), 'datasets', "SEED_IV", 'npz', "Preprocessed")

if METHOD == 'noseg':
    save_datas_noseg(src_dir, join(saved_dir, 'no_seg'))

elif METHOD == 'seg':
    save_datas_seg(WINDOW, STRIDE, src_dir,join(saved_dir, 'seg'))

elif METHOD == 'PSD':
    save_datas_seg_PSD(WINDOW, STRIDE, src_dir, join(saved_dir, 'seg_PSD'))

elif METHOD == 'DE': # DE calculation takes a time. be careful
    save_datas_seg_DE(WINDOW, STRIDE, src_dir, join(saved_dir, 'seg_DE'))

# -----------------------------------------check---------------------------------------------------
# Save the bar graph of the number of labels per class
# from utils.tools import getFromnpz, plot_VA

# load data
# saved_dir = join(DATAS, 'npz', 'Preprocessed','seg_DE')

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
import os
from os.path import join
import numpy as np
from scipy import io
import re
import argparse

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
                    y.append([labels[trial_id], subidx])
                    idx += stride
        x = np.array(x)
        x = np.array(x, dtype='float16')
        np.nan_to_num(x, copy=False)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subidx).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')

from utils.transform import BandDifferentialEntropy
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
                    x.append(bde.apply(seg))
                    y.append([labels[trial_id], subidx])
                    idx += stride
        x = np.array(x)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subidx).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')

    
from utils.transform import BandPowerSpectralDensity
def save_datas_seg_PSD(window, stride, data_dir, saved_dir):
    print('Segmentation with PSD x: (samples, 62, 4), y: (samples, 2)')

    psd = BandPowerSpectralDensity()

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
                    x.append(psd.apply(seg))
                    y.append([labels[trial_id], subidx])
                    idx += stride
        x = np.array(x)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subidx).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')


# -----------------------------------------save-data---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, default="/mnt/data")
parser.add_argument("--window", type=int, default=400)
parser.add_argument("--stride", type=int, default=200)
parser.add_argument("--method", type=str, default="seg", help='seg, PSD, DE')
args = parser.parse_args()

SRC = args.src_dir
WINDOW = args.window
STRIDE = args.stride
METHOD = args.method
src_dir = join(SRC, 'SEED', 'Preprocessed_EEG')
saved_dir = join(os.getcwd(), 'datasets', "SEED", 'npz', "Preprocessed")

if METHOD == 'seg':
    save_datas_seg(WINDOW, STRIDE, src_dir,join(saved_dir, 'seg'))

elif METHOD == 'PSD':
    save_datas_seg_PSD(WINDOW, STRIDE, src_dir, join(saved_dir, 'seg_PSD'))

elif METHOD == 'DE':
    save_datas_seg_DE(WINDOW, STRIDE, src_dir, join(saved_dir, 'seg_DE'))

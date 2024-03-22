import os
from os.path import join
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
                    y.append([session_label[session-1][trial_id], subnums[subidx]])
                    idx += stride
        x = np.array(x, dtype='float16')
        np.nan_to_num(x, copy=False)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subnums[subidx]).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')

from utils.transform import BandDifferentialEntropy
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
                    y.append([session_label[session-1][trial_id], subnums[subidx]])
                    idx += stride
        x = np.array(x)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
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
                    y.append([session_label[session-1][trial_id], subnums[subidx]])
                    idx += stride
        x = np.array(x)
        y = np.array(y)
        
        print(f'EEG:{x.shape} label:{y.shape}')
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, str(subnums[subidx]).zfill(2)), x=x, y=y) 
    print(f'saved in {saved_dir}')
# -----------------------------------------save data-------------------------------------------------
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
src_dir = join(SRC, 'SEED_IV', 'eeg_raw_data')
saved_dir = join(os.getcwd(), 'datasets', "SEED_IV", 'npz', "Preprocessed")

if METHOD == 'seg':
    save_datas_seg(WINDOW, STRIDE, src_dir,join(saved_dir, 'seg'))

elif METHOD == 'PSD':
    save_datas_seg_PSD(WINDOW, STRIDE, src_dir, join(saved_dir, 'seg_PSD'))

elif METHOD == 'DE':
    save_datas_seg_DE(WINDOW, STRIDE, src_dir, join(saved_dir, 'seg_DE'))

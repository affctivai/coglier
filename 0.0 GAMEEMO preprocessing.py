import os
from os.path import join
import pandas as pd
import numpy as np
from utils.constant import *
import argparse

def save_datas_seg(src, window, stride, emotions, channels ,sublist, save_path):
    print('Segmentation x: (samples, channels, winodw size), y: (samples, 2) # [label, subID]')
    for subnum in sublist:
        print('sub ID:',subnum, end=' ')
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for emo in emotions:
            EEG_path = join(src, f'(S{subnum})','Preprocessed EEG Data','.csv format',f'S{subnum}{emo}'+'AllChannels.csv')
            csv = pd.read_csv(EEG_path, usecols=channels)
            data = csv.to_numpy()

            SAM_path = join(src, f'(S{subnum})','SAM Ratings',f'{emo}.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()

            valence, arousal = int(label[1]), int(label[-1])

            n = len(data)
            idx = 0
            while idx + window < n:
                seg = data[idx : idx + window]
                seg = seg.swapaxes(0, 1)
                sub_x.append(seg)
                sub_y.append([label, int(subnum)])
                sub_v.append([valence-1, int(subnum)])
                sub_a.append([arousal-1, int(subnum)])
                idx += stride

        sub_x = np.array(sub_x);    sub_y = np.array(sub_y);    sub_v = np.array(sub_v);    sub_a = np.array(sub_a);
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        os.makedirs(save_path, exist_ok=True)
        np.savez(join(save_path, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {save_path}')

def save_datas_seg_DE(src, window, stride, emotions, channels ,sublist, save_path):
    from utils.transform import BandDifferentialEntropy
    print('Segmentation with DE x: (samples, channels, 4 freq), y: (samples, 2) # [label, subID]')
    for subnum in sublist:
        print('sub ID:',subnum, end=' ')
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for emo in emotions:
            EEG_path = join(src, f'(S{subnum})','Preprocessed EEG Data','.csv format',f'S{subnum}{emo}'+'AllChannels.csv')
            csv = pd.read_csv(EEG_path, usecols=channels)
            data = csv.to_numpy()
            SAM_path = join(src, f'(S{subnum})','SAM Ratings',f'{emo}.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()
            valence, arousal = int(label[1]), int(label[-1])

            n = len(data)
            idx = 0
            while idx + window < n:
                seg = data[idx : idx + window]
                seg = seg.swapaxes(0, 1)

                bde = BandDifferentialEntropy()
                sub_x.append(bde.apply(seg))

                sub_y.append([label, int(subnum)])
                sub_v.append([valence-1, int(subnum)])
                sub_a.append([arousal-1, int(subnum)])
                idx += stride

        sub_x = np.array(sub_x);    sub_y = np.array(sub_y);    sub_v = np.array(sub_v);    sub_a = np.array(sub_a);
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        os.makedirs(save_path, exist_ok=True)
        np.savez(join(save_path, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {save_path}')

def save_datas_seg_PSD(src, window, stride, emotions, channels ,sublist, save_path):
    from utils.transform import BandPowerSpectralDensity
    print('Segmentation with PSD x: (samples, channels, 4 freq), y: (samples, 2) # [label, subID]')
    for subnum in sublist:
        print('sub ID:',subnum, end=' ')
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for emo in emotions:
            EEG_path = join(src, f'(S{subnum})','Preprocessed EEG Data','.csv format',f'S{subnum}{emo}'+'AllChannels.csv')
            csv = pd.read_csv(EEG_path, usecols=channels)
            data = csv.to_numpy()
            SAM_path = join(src, f'(S{subnum})','SAM Ratings',f'{emo}.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()
            valence, arousal = int(label[1]), int(label[-1])

            n = len(data)
            idx = 0
            while idx + window < n:
                seg = data[idx : idx + window]
                seg = seg.swapaxes(0, 1)

                psd = BandPowerSpectralDensity()
                sub_x.append(psd.apply(seg))

                sub_y.append([label, int(subnum)])
                sub_v.append([valence-1, int(subnum)])
                sub_a.append([arousal-1, int(subnum)])
                idx += stride

        sub_x = np.array(sub_x);    sub_y = np.array(sub_y);    sub_v = np.array(sub_v);    sub_a = np.array(sub_a);
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        os.makedirs(save_path, exist_ok=True)
        np.savez(join(save_path, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {save_path}')

# -----------------------------------------main---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, default="/mnt/data")
parser.add_argument("--window", type=int, default=256)
parser.add_argument("--stride", type=int, default=128)
parser.add_argument("--method", type=str, default="seg", help='seg, PSD, DE')
args = parser.parse_args()

SRC = args.src_dir
WINDOW = args.window
STRIDE = args.stride
METHOD = args.method

SUBLIST = [str(i).zfill(2) for i in range(1, GAMEEMO_SUBNUM + 1)]
EMOS = ['G1', 'G2', 'G3', 'G4']
src_dir = join(SRC, 'GAMEEMO')
save_dir = join(os.getcwd(), 'datasets', 'GAMEEMO', 'npz', 'Preprocessed')

if METHOD == 'seg':
    save_datas_seg(src_dir, WINDOW, STRIDE, EMOS, GAMEEMO_CHLS, SUBLIST, join(save_dir, 'seg'))

elif METHOD == 'PSD':
    save_datas_seg_PSD(src_dir, WINDOW, STRIDE, EMOS, GAMEEMO_CHLS, SUBLIST, join(save_dir, 'seg_PSD'))

elif METHOD == 'DE':
    save_datas_seg_DE(src_dir, WINDOW, STRIDE, EMOS, GAMEEMO_CHLS, SUBLIST, join(save_dir, 'seg_DE'))

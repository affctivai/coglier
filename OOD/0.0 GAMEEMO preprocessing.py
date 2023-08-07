import os
from os.path import join
import pandas as pd
import numpy as np
from utils.constant import *
import argparse


# No segmentation x: (4, 14, 38252), y: (4,)
def save_datas_noseg(src, emotions, channels, sublist, save_path):
    print('No segmetation. x: (4,14,38252) y: (4,)')
    for subnum in sublist:
        print('sub ID:',subnum, end=' ')
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for emo in emotions:
            # ex) S01G1AllChannels.csv
            EEG_path = join(src, f'(S{subnum})','Preprocessed EEG Data','.csv format',f'S{subnum}{emo}'+'AllChannels.csv')
            csv = pd.read_csv(EEG_path, usecols=channels)
            data = csv.to_numpy()
            data = data.swapaxes(0, 1) # (time, channel) -> (channel, time)

            # ex) G1.txt
            SAM_path = join(src, f'(S{subnum})','SAM Ratings',f'{emo}.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()

            # V1A1 1~9
            valence, arousal = int(label[1]), int(label[-1])

            sub_x.append(data); sub_y.append(label);    sub_v.append(valence);  sub_a.append(arousal);

        sub_x = np.array(sub_x);    sub_y = np.array(sub_y);    sub_v = np.array(sub_v);    sub_a = np.array(sub_a);
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        # save sub_x, sub_y, sub_a, sub_v
        os.makedirs(save_path, exist_ok=True)
        np.savez(join(save_path, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {save_path}')

# segmentation x: (samples, channels, window size), y: (samples, 2)  ## [label, subID] 
## if window: 256, stride: 128 -> x: (1188, 14, 256), y: (1188, 2)
def save_datas_seg(src, window, stride, emotions, channels ,sublist, save_path):
    print('Segmentation x: (samples, channels, winodw size), y: (samples, 2) # [label, subID]')
    for subnum in sublist:
        print('sub ID:',subnum, end=' ')
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for emo in emotions:
            # ex) S01G1AllChannels.csv
            EEG_path = join(src, f'(S{subnum})','Preprocessed EEG Data','.csv format',f'S{subnum}{emo}'+'AllChannels.csv')
            csv = pd.read_csv(EEG_path, usecols=channels)
            data = csv.to_numpy() # (time, channel)

            # ex) G1.txt
            SAM_path = join(src, f'(S{subnum})','SAM Ratings',f'{emo}.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()

            # V1A1 1~9
            valence, arousal = int(label[1]), int(label[-1])

            # Segmentation
            n = len(data)
            idx = 0
            while idx + window < n:
                seg = data[idx : idx + window]
                seg = seg.swapaxes(0, 1) # (channel, time)
                sub_x.append(seg)
                sub_y.append([label, int(subnum)]) # label, subID
                sub_v.append([valence-1, int(subnum)])  # label, subID
                sub_a.append([arousal-1, int(subnum)])  # label, subID
                idx += stride

        sub_x = np.array(sub_x);    sub_y = np.array(sub_y);    sub_v = np.array(sub_v);    sub_a = np.array(sub_a);
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        # save sub_x, sub_y, sub_a, sub_v
        os.makedirs(save_path, exist_ok=True)
        np.savez(join(save_path, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {save_path}')

# segmentation -> DE(BandDifferentialEntropy)
# x: (samples, channels, winodw size) -> (samples, channels, 4(frequency))   y: (samples, 2) ## [label, subID]
## if window: 256, stride: 128 -> x: (1188, 14, 4), y: (1188, 2)
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
            data = csv.to_numpy() # (time, channel)
            SAM_path = join(src, f'(S{subnum})','SAM Ratings',f'{emo}.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()
            valence, arousal = int(label[1]), int(label[-1])

            # Segmentation
            n = len(data)
            idx = 0
            while idx + window < n:
                seg = data[idx : idx + window]
                seg = seg.swapaxes(0, 1) # (channel, time)

                # Preprocessing
                bde = BandDifferentialEntropy()
                sub_x.append(bde.apply(seg))

                sub_y.append([label, int(subnum)]) # label, subID
                sub_v.append([valence-1, int(subnum)])  # label, subID
                sub_a.append([arousal-1, int(subnum)])  # label, subID
                idx += stride

        sub_x = np.array(sub_x);    sub_y = np.array(sub_y);    sub_v = np.array(sub_v);    sub_a = np.array(sub_a);
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        # save sub_x, sub_y, sub_a, sub_v
        os.makedirs(save_path, exist_ok=True)
        np.savez(join(save_path, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {save_path}')

# segmentation -> PSD(power spectral density)
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
            data = csv.to_numpy() # (time, channel)
            SAM_path = join(src, f'(S{subnum})','SAM Ratings',f'{emo}.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()
            valence, arousal = int(label[1]), int(label[-1])

            # Segmentation
            n = len(data)
            idx = 0
            while idx + window < n:
                seg = data[idx : idx + window]
                seg = seg.swapaxes(0, 1) # (channel, time)

                # Preprocessing
                psd = BandPowerSpectralDensity()
                sub_x.append(psd.apply(seg))

                sub_y.append([label, int(subnum)]) # label, subID
                sub_v.append([valence-1, int(subnum)])  # label, subID
                sub_a.append([arousal-1, int(subnum)])  # label, subID
                idx += stride

        sub_x = np.array(sub_x);    sub_y = np.array(sub_y);    sub_v = np.array(sub_v);    sub_a = np.array(sub_a);
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        # save sub_x, sub_y, sub_a, sub_v
        os.makedirs(save_path, exist_ok=True)
        np.savez(join(save_path, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {save_path}')

# -----------------------------------------main---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, default="/mnt/data/GAMEEMO") # source data folder location
parser.add_argument("--save_dir", type=str, default="/mnt/data/research_EG") # path to save preprocessed data(.npz format)
parser.add_argument("--window", type=int, default=256)
parser.add_argument("--stride", type=int, default=128)
parser.add_argument("--method", type=str, default="seg", help='noseg, seg, PSD, DE')
args = parser.parse_args()

SRC = args.src_dir
SAVE = args.save_dir
WINDOW = args.window
STRIDE = args.stride
METHOD = args.method

SUBLIST = [str(i).zfill(2) for i in range(1, GAMEEMO_SUBNUM + 1)] # '01', '02', '03', ...
EMOS = ['G1', 'G2', 'G3', 'G4'] # 'boring', 'calm', 'horror', 'funny'
save_dir = join(args.save_dir, 'GAMEEMO_npz', 'Preprocessed')

if METHOD == 'noseg':
    save_datas_noseg(SRC, EMOS, GAMEEMO_CHLS, SUBLIST, join(save_dir, 'noseg'))

elif METHOD == 'seg':
    save_datas_seg(SRC, WINDOW, STRIDE, EMOS, GAMEEMO_CHLS, SUBLIST, join(save_dir, 'seg'))

elif METHOD == 'PSD':
    save_datas_seg_PSD(SRC, WINDOW, STRIDE, EMOS, GAMEEMO_CHLS, SUBLIST, join(save_dir, 'seg_PSD'))

elif METHOD == 'DE': # DE calculation takes a time. be careful
    save_datas_seg_DE(SRC, WINDOW, STRIDE, EMOS, GAMEEMO_CHLS, SUBLIST, join(save_dir, 'seg_DE'))

# -----------------------------------------check---------------------------------------------------
# Save the bar graph of the number of labels per class
def saveBarGraph(window, stride):
    from utils.tools import getFromnpz, plot_VA

    datas_v, targets_v = getFromnpz(join(save_dir, 'seg'), SUBLIST, out=False, cla='v')
    datas_a, targets_a = getFromnpz(join(save_dir, 'seg'), SUBLIST, out=False, cla='a')
    vals, count_v = np.unique(targets_v[:, 0], return_counts=True)
    aros, count_a = np.unique(targets_a[:, 0], return_counts=True)

    # subIDs, countss_v = np.unique(targets_v[:, 1], return_counts=True)
    # subIDs, countss_a = np.unique(targets_a[:, 1], return_counts=True)

    print(f'data_v shape: {datas_v.shape} target_v shape: {targets_v.shape}')
    print(f'data_a shape: {datas_a.shape} target_a shape: {targets_a.shape}')
    print(f'valence {vals} \t {count_v}')
    print(f'arousal {aros} \t {count_a}')
    # print(f'Num of data per subject {subIDs} \t {countss_v}') # subIDs

    file_name = f'bar_graph_w{window}_s{stride}'
    plot_VA(vals, count_v, aros, count_a, path=join(args.save_dir,'GAMEEMO_npz', file_name))

# saveBarGraph(WINDOW, STRIDE)

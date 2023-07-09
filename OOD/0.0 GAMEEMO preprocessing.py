import os
from os.path import join
import pandas as pd
import numpy as np

# No segmentation x: (4, 14, 38252), y: (4,)
def save_datas_noseg(emotions, channels, sublist, saved_dir):
    print('No segmetation. x: (4,14,38252) y: (4,)')
    for subnum in sublist:
        print('sub ID:',subnum, end=' ')
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for emo in emotions:
            # ex) S01G1AllChannels.csv
            EEG_path = join(DATA, '(S' + subnum + ')', 'Preprocessed EEG Data', '.csv format',\
                                'S' + subnum + emo + 'AllChannels.csv')
            csv = pd.read_csv(EEG_path, usecols=channels)
            data = csv.to_numpy()
            data = data.swapaxes(0, 1) # (time, channel) -> (channel, time)

            # ex) G1.txt
            SAM_path = join(DATA, '(S' + subnum + ')', 'SAM Ratings', emo+'.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()

            # V1A1 1~9
            valence, arousal = int(label[1]), int(label[-1])

            # 5 is the reference point
            if arousal > 5:
                if valence > 5: label = 1 # HAPV
                elif valence < 5: label = 2 # HANV
                else: label = 0 # val==5
            elif arousal < 5:
                if valence > 5: label = 3 #LAPV
                elif valence < 5: label = 4 # LANV
                else: label = 0 # val==5
            else: label = 0 # aro==5

            sub_x.append(data)
            sub_y.append(label)
            sub_v.append(valence)
            sub_a.append(arousal)

        sub_x = np.array(sub_x)
        sub_y = np.array(sub_y)
        sub_v = np.array(sub_v)
        sub_a = np.array(sub_a)
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        # save sub_x, sub_y, sub_a, sub_v
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {saved_dir}')

# segmentation x: (samples, 14, segment size), y: (samples, 2)  ## [label, subID]
## if window: 256, stride: 128 -> x: (1188, 14, 256), y: (1188, 2)
def save_datas_seg(window, stride, emotions, channels ,sublist, saved_dir):
    print('Segmentation x: (samples, 14, segment size), y: (samples, 2)')
    for subnum in sublist:
        print('sub ID:',subnum, end=' ')
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for emo in emotions:
            # ex) S01G1AllChannels.csv
            EEG_path = join(DATA, '(S' + subnum + ')', 'Preprocessed EEG Data', '.csv format',\
                                'S' + subnum + emo + 'AllChannels.csv')
            csv = pd.read_csv(EEG_path, usecols=channels)
            data = csv.to_numpy() # (time, channel)

            # ex) G1.txt
            SAM_path = join(DATA, '(S' + subnum + ')', 'SAM Ratings', emo+'.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()

            # V1A1 1~9
            valence, arousal = int(label[1]), int(label[-1])

            # 5 is the reference point
            if arousal > 5:
                if valence > 5: label = 1  # HAPV
                elif valence < 5: label = 2  # HANV
                else: label = 0 # val==5
            elif arousal < 5:
                if valence > 5: label = 3  # LAPV
                elif valence < 5: label = 4  # LANV
                else: label = 0 # val==5
            else: label = 0 # aro==5

            # if label != 0: # Not (arousal==5 or valence==5)

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

        sub_x = np.array(sub_x)
        sub_y = np.array(sub_y)
        sub_v = np.array(sub_v)
        sub_a = np.array(sub_a)
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        # save sub_x, sub_y, sub_a, sub_v
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {saved_dir}')

# segmentation -> DE(BandDifferentialEntropy)
# x: (samples, 14, segment size) -> (samples, 14, 4(frequency))   y: (samples, 2) ## [label, subID]
## if window: 256, stride: 128 -> x: (1188, 14, 4), y: (1188, 2)
def save_datas_seg_DE(window, stride, emotions, channels ,sublist, saved_dir):
    from utils.transform import BandDifferentialEntropy
    print('Segmentation with DE x: (samples, 14, 4), y: (samples, 2)')
    for subnum in sublist:
        print('sub ID:',subnum, end=' ')
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for emo in emotions:
            # ex) S01G1AllChannels.csv
            EEG_path = join(DATA, '(S' + subnum + ')', 'Preprocessed EEG Data', '.csv format',\
                                'S' + subnum + emo + 'AllChannels.csv')
            csv = pd.read_csv(EEG_path, usecols=channels)
            data = csv.to_numpy() # (time, channel)

            # ex) G1.txt
            SAM_path = join(DATA, '(S' + subnum + ')', 'SAM Ratings', emo+'.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()

            # V1A1 1~9
            valence, arousal = int(label[1]), int(label[-1])

            # 5 is the reference point
            if arousal > 5:
                if valence > 5: label = 1  # HAPV
                elif valence < 5: label = 2  # HANV
                else: label = 0
            elif arousal < 5:
                if valence > 5: label = 3  # LAPV
                elif valence < 5: label = 4  # LANV
                else: label = 0
            else: label = 0

            n = len(data)
            idx = 0
            while idx + window < n:
                seg = data[idx : idx + window]
                seg = seg.swapaxes(0, 1) # (channel, time)

                # preprocessing
                bde = BandDifferentialEntropy()
                sub_x.append(bde.apply(seg))

                sub_y.append([label, int(subnum)]) # label, subID
                sub_v.append([valence-1, int(subnum)])  # label, subID
                sub_a.append([arousal-1, int(subnum)])  # label, subID
                idx += stride

        sub_x = np.array(sub_x)
        sub_y = np.array(sub_y)
        sub_v = np.array(sub_v)
        sub_a = np.array(sub_a)
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        # save sub_x, sub_y, sub_a, sub_v
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {saved_dir}')

# segmentation -> PSD(power spectral density)
def save_datas_seg_PSD(window, stride, emotions, channels ,sublist, saved_dir):
    from utils.transform import BandPowerSpectralDensity
    print('Segmentation with PSD x: (samples, 14, 4), y: (samples, 2)')
    for subnum in sublist:
        print('sub ID:',subnum, end=' ')
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for emo in emotions:
            # ex) S01G1AllChannels.csv
            EEG_path = join(DATA, '(S' + subnum + ')', 'Preprocessed EEG Data', '.csv format',\
                                'S' + subnum + emo + 'AllChannels.csv')
            csv = pd.read_csv(EEG_path, usecols=channels)
            data = csv.to_numpy() # (time, channel)

            # ex) G1.txt
            SAM_path = join(DATA, '(S' + subnum + ')', 'SAM Ratings', emo+'.txt')
            label = open(SAM_path, 'r')
            label = label.readline().strip()

            # V1A1 1~9
            valence, arousal = int(label[1]), int(label[-1])

            # 5 is the reference point
            if arousal > 5:
                if valence > 5: label = 1  # HAPV
                elif valence < 5: label = 2  # HANV
                else: label = 0
            elif arousal < 5:
                if valence > 5: label = 3  # LAPV
                elif valence < 5: label = 4  # LANV
                else: label = 0
            else: label = 0

            n = len(data)
            idx = 0
            while idx + window < n:
                seg = data[idx : idx + window]
                seg = seg.swapaxes(0, 1) # (channel, time)

                # preprocessing
                psd = BandPowerSpectralDensity()
                sub_x.append(psd.apply(seg))

                sub_y.append([label, int(subnum)]) # label, subID
                sub_v.append([valence-1, int(subnum)])  # label, subID
                sub_a.append([arousal-1, int(subnum)])  # label, subID
                idx += stride

        sub_x = np.array(sub_x)
        sub_y = np.array(sub_y)
        sub_v = np.array(sub_v)
        sub_a = np.array(sub_a)
        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')

        # save sub_x, sub_y, sub_a, sub_v
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {saved_dir}')
# -----------------------------------------main---------------------------------------------------
# source data folder location
DATAS = join("C:\\", "Users", "LAPTOP", "jupydir", "DATAS")
DATA = join(DATAS, 'GAMEEMO_data')

sublists = os.listdir(DATA) # '(S01)', '(S02)', '(S03)', ...
sublists = [subname[2:-1] for subname in sublists] # # '01', '02', '03', ...

CHLS = ['AF3','AF4','F3','F4','F7','F8','FC5','FC6','O1','O2','P7','P8','T7','T8'] # 14 channels
EMOS = ['G1', 'G2', 'G3', 'G4'] # 'boring', 'calm', 'horror', 'funny'

WINDOW = 128 * 2
STRIDE = 128
# -----------------------------------------save data-------------------------------------------------
# path to save preprocessed data(.npz format)
saved_dir = join(DATAS, 'GAMEEMO_npz', 'Preprocessed')

# methods
# save_datas_noseg(EMOS, CHLS, sublists, join(saved_dir, 'no_seg'))
# save_datas_seg(WINDOW, STRIDE, EMOS, CHLS ,sublists, join(saved_dir, 'seg'))

## DE calculation takes a time. be careful
# save_datas_seg_DE(WINDOW, STRIDE, EMOS, CHLS ,sublists, join(saved_dir, 'seg_DE'))

# PSD
save_datas_seg_PSD(WINDOW, STRIDE, EMOS, CHLS ,sublists, join(saved_dir, 'seg_PSD'))
# -----------------------------------------check---------------------------------------------------
# # Save the bar graph of the number of labels per class
# from utils.tools import getFromnpz, plot_VA
#
# # load data
# dir_name = 'seg_PSD'
# saved_dir = join(DATAS, 'GAMEEMO_npz', dir_name)
#
# datas_v, targets_v = getFromnpz(saved_dir, sublists, out=False, cla='v')
# datas_a, targets_a = getFromnpz(saved_dir, sublists, out=False, cla='a')
# vals, count_v = np.unique(targets_v[:, 0], return_counts=True)
# aros, count_a = np.unique(targets_a[:, 0], return_counts=True)
#
# # subIDs, countss_v = np.unique(targets_v[:, 1], return_counts=True)
# # subIDs, countss_a = np.unique(targets_a[:, 1], return_counts=True)
#
# print(f'data_v shape: {datas_v.shape} target_v shape: {targets_v.shape}')
# print(f'data_a shape: {datas_a.shape} target_a shape: {targets_a.shape}')
# print(f'valence {vals} \t {count_v}')
# print(f'arousal {aros} \t {count_a}')
# # print(f'Num of data per subject {subIDs} \t {countss_v}') # subIDs
#
# plot_VA(vals, count_v, aros, count_a, path=join(DATAS,'GAMEEMO_npz', dir_name))
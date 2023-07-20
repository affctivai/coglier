import os
from os.path import join
import pandas as pd
import numpy as np
from scipy import io
import re

def save_datas_noseg(sublist, data_dir, saved_dir):
    print('No Segmentation x: (samples, 62, segment size), y: (samples, 2)')

    for subnum in sublist:
        print(subnum, end=' ')
        path = os.path.join(data_dir, "s%02d.mat" % int(subnum))
        datas = io.loadmat(path)
        data = datas['data']
        labels = datas['labels']
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for i in range(40):

            # V1A1 1~9
            valence, arousal = labels[i, 0], labels[i, 1]

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

            seg = data[i, :32, :]
            sub_x.append(seg)
            sub_y.append([label, int(subnum)]) # label, subID
            sub_v.append([int(valence)-1, int(subnum)])  # label, subID
            sub_a.append([int(arousal)-1, int(subnum)])  # label, subID
        sub_x = np.array(sub_x)
        sub_y = np.array(sub_y)
        sub_v = np.array(sub_v)
        sub_a = np.array(sub_a)

        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {saved_dir}')

# segmentation x: (samples, 14, segment size), y: (samples, 2)  ## [label, subID]
## if window: 256, stride: 128 -> x: (15718, 62, 256), y: (15718, 2)
def save_datas_seg(window, stride, sublist, data_dir, saved_dir):
    print('Segmentation x: (samples, 62, segment size), y: (samples, 2)')

    for subnum in sublist:
        print(subnum, end=' ')
        path = os.path.join(data_dir, "s%02d.mat" % int(subnum))
        datas = io.loadmat(path)
        data = datas['data']
        labels = datas['labels']
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for i in range(40):

            # V1A1 1~9
            valence, arousal = labels[i, 0], labels[i, 1]

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

            idx = 0
            time_size = 8064
            while idx + window < time_size:
                seg = data[i, :32, idx : idx+window]
                sub_x.append(seg)
                sub_y.append([label, int(subnum)]) # label, subID
                sub_v.append([int(valence)-1, int(subnum)])  # label, subID
                sub_a.append([int(arousal)-1, int(subnum)])  # label, subID
                idx += stride
        sub_x = np.array(sub_x)
        sub_y = np.array(sub_y)
        sub_v = np.array(sub_v)
        sub_a = np.array(sub_a)

        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {saved_dir}')

from utils.transform import BandDifferentialEntropy
# segmentation -> DE(BandDifferentialEntropy)
# x: (samples, 62, segment size) -> (samples, 62, 4(frequency))   y: (samples, 2) ## [label, subID]
## if window: 256, stride: 128 -> x: (15718, 62, 4), y: (15718, 2)
def save_datas_seg_DE(window, stride, sublist, data_dir, saved_dir):
    print('Segmentation with DE x: (samples, 62, 4), y: (samples, 2)')

    bde = BandDifferentialEntropy()
    for subnum in sublist:
        print(subnum, end=' ')
        path = os.path.join(data_dir, "s%02d.mat" % int(subnum))
        datas = io.loadmat(path)
        data = datas['data']
        labels = datas['labels']
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for i in range(40):

            # V1A1 1~9
            valence, arousal = labels[i, 0], labels[i, 1]

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

            idx = 0
            time_size = 8064
            while idx + window < time_size:
                seg = data[i, :32, idx : idx+window]
                sub_x.append(bde.apply(seg))
                sub_y.append([label, int(subnum)]) # label, subID
                sub_v.append([int(valence)-1, int(subnum)])  # label, subID
                sub_a.append([int(arousal)-1, int(subnum)])  # label, subID
                idx += stride
        sub_x = np.array(sub_x)
        sub_y = np.array(sub_y)
        sub_v = np.array(sub_v)
        sub_a = np.array(sub_a)

        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {saved_dir}')

from utils.transform import BandPowerSpectralDensity
def save_datas_seg_PSD(window, stride, sublist, data_dir, saved_dir):
    print('Segmentation with PSD x: (samples, 62, 4), y: (samples, 2)')

    psd = BandPowerSpectralDensity()
    for subnum in sublist:
        print(subnum, end=' ')
        path = os.path.join(data_dir, "s%02d.mat" % int(subnum))
        datas = io.loadmat(path)
        data = datas['data']
        labels = datas['labels']
        sub_x, sub_y = [], []
        sub_v, sub_a = [], []
        for i in range(40):

            # V1A1 1~9
            valence, arousal = labels[i, 0], labels[i, 1]

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

            idx = 0
            time_size = 8064
            while idx + window < time_size:
                seg = data[i, :32, idx : idx+window]
                sub_x.append(psd.apply(seg))
                sub_y.append([label, int(subnum)]) # label, subID
                sub_v.append([int(valence)-1, int(subnum)])  # label, subID
                sub_a.append([int(arousal)-1, int(subnum)])  # label, subID
                idx += stride
        sub_x = np.array(sub_x)
        sub_y = np.array(sub_y)
        sub_v = np.array(sub_v)
        sub_a = np.array(sub_a)

        print(f'EEG:{sub_x.shape} aro:{sub_a.shape} val:{sub_v.shape}')
        # 저장 폴더에 subject 별로 npz 파일 생성됨
        os.makedirs(saved_dir, exist_ok=True)
        np.savez(join(saved_dir, subnum), x=sub_x, y=sub_y, v=sub_v, a=sub_a)
    print(f'saved in {saved_dir}')


# -----------------------------------------main---------------------------------------------------
# source data folder location
DATAS = join(os.getcwd(),"datasets", "DEAP")
DATA = os.path.join(os.getcwd(),".." , "..","..", "..", "dataset", "DEAP", "data_preprocessed_matlab")

WINDOW = 128 * 2
STRIDE = 128

sub_list = os.listdir(DATA)
sub_list = [subname[1:-4] for subname in sub_list]
print(sub_list) # subject ID list
# -----------------------------------------save data-------------------------------------------------
# path to save preprocessed data(.npz format)
saved_dir = join(DATAS, "npz", 'Preprocessed')

# There are 2 methods
# save_datas_noseg(sub_list, DATA, join(saved_dir, 'no_seg'))
# save_datas_seg(WINDOW, STRIDE, sub_list, DATA, join(saved_dir, 'seg_raw'))
## DE calculation takes a time. be careful
# save_datas_seg_DE(WINDOW, STRIDE, sub_list, DATA, join(saved_dir, 'seg_DE'))
save_datas_seg_PSD(WINDOW, STRIDE, sub_list, DATA, join(saved_dir, 'seg_PSD'))

# -----------------------------------------check---------------------------------------------------
# Save the bar graph of the number of labels per class
# from utils.tools import getFromnpz, plot_VA

# load data
# saved_dir = join(DATAS, 'DEAP_npz', 'seg_raw')

# datas_v, targets_v = getFromnpz(saved_dir, sub_list, out=False, cla='v')
# datas_a, targets_a = getFromnpz(saved_dir, sub_list, out=False, cla='a')
# vals, count_v = np.unique(targets_v[:, 0], return_counts=True)
# aros, count_a = np.unique(targets_a[:, 0], return_counts=True)

# subIDs, countss_v = np.unique(targets_v[:, 1], return_counts=True)
# subIDs, countss_a = np.unique(targets_a[:, 1], return_counts=True)

# print(f'data_v shape: {datas_v.shape} target_v shape: {targets_v.shape}')
# print(f'data_a shape: {datas_a.shape} target_a shape: {targets_a.shape}')
# print(f'valence {vals} \t {count_v}')
# print(f'arousal {aros} \t {count_a}')
# print(f'Num of data per subject {subIDs} \t {countss_v}') # subIDs

# plot_VA(vals, count_v, aros, count_a, path=join(DATAS,'DEAP_npz','seg_DE.png'))
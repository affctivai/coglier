import os
from os.path import join, split
import random
import numpy as np

from utils.tools import getFromnpz, getFromnpz_
from utils.transform import make_grid
from sklearn.model_selection import train_test_split

def seed(s):
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
SEED = 42
seed(SEED)

def save_dataset(folder, names, mode, x, y, out=True):
    file_name = f'{names}_{mode}'
    np.savez(join(folder, file_name), X=x, Y=y)
    if out: print(f'saved in {folder}')

# Subject Independent
def make_dataset_SI(src, sublists, label, save_folder):
    datas, targets = getFromnpz(src, sublists, out=True, cla=label)
    labels, countsl = np.unique(targets[:, 0], return_counts=True)
    subIDs, countss = np.unique(targets[:, 1], return_counts=True)

    print(f'data shape: {datas.shape} target shape: {targets.shape}')
    print(f'label {label} count {labels} \t {countsl}') # labels
    print(f'subID count {subIDs} \t {countss}\n') # subIDs
    # for CNN model 3D->4D  ## (samples, 14, 4 bands) -> (samples, 4 bands, 9, 9)
    datas = make_grid(datas)
    print(f'data shape after making 4D(samples, 4freq, 9x9): {datas.shape}')

    # Make Dataset
    ## train 80 : valid 10 : test 10
    X_train, X, Y_train, Y = train_test_split(datas, targets, test_size=0.2, stratify=targets, random_state=SEED)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=SEED)
    print(f'num of train: {len(Y_train)} \t num of valid: {len(Y_valid)} \t num of test: {len(Y_test)}\n')

    ## save train, valid, test
    os.makedirs(save_folder, exist_ok=True)
    names = f'{split(src)[-1]}_{label}' # seg_DE_v

    save_dataset(save_folder, names, 'train', X_train, Y_train)
    save_dataset(save_folder, names, 'valid', X_valid, Y_valid)
    save_dataset(save_folder, names, 'test', X_test, Y_test)

# Subject dependent
def make_dataset_SD(src, sublists, label, save_folder):
    for sub in sublists:
        datas, targets = getFromnpz_(src, sub, out=True, cla=label)
        labels, countsl = np.unique(targets[:, 0], return_counts=True)
        print(f'label {label} count {labels} \t {countsl}')  # labels
        # for CNN model 3D->4D  ## (samples, 14, 4 bands) -> (samples, 4 bands, 9, 9)
        datas = make_grid(datas)
        # Make Dataset  ## train 80 : valid 10 : test 10
        X_train, X, Y_train, Y = train_test_split(datas, targets, test_size=0.2, stratify=targets, random_state=SEED)
        X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=SEED)

        ## save train, valid, test
        save_folder_sub = join(save_folder, sub)
        os.makedirs(save_folder_sub, exist_ok=True)
        names = f'{split(src)[-1]}_{label}'  # seg_DE_v

        save_dataset(save_folder_sub, names, 'train', X_train, Y_train, out=False)
        save_dataset(save_folder_sub, names, 'valid', X_valid, Y_valid, out=False)
        save_dataset(save_folder_sub, names, 'test', X_test, Y_test, out=False)

    print(f'data shape after making 4D(samples, 4freq, 9x9): {datas.shape}')
    print(f'num of train: {len(Y_train)} \t num of valid: {len(Y_valid)} \t num of test: {len(Y_test)}\n')

# High Low dataset
def make_dataset_HL(src, ranks, cut, label, save_folder):
    ranks = [str(sub).zfill(2) for sub in ranks]
    # cut = int(28 * cut_rate)
    higs = ranks[: cut]
    lows = ranks[cut :]
    print(label, 'high', len(higs), '명', higs)
    print(label, 'low ', len(lows), '명', lows)

    # make High dataset. train 80 : valid 10 : test 10
    datas_h, targets_h = getFromnpz(src, higs, out=True, cla=label)
    datas_h = make_grid(datas_h)

    X_train, X, Y_train, Y = train_test_split(datas_h, targets_h, test_size=0.2, stratify=targets_h, random_state=SEED)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=SEED)
    print(f'High train: {len(Y_train)} \t High valid: {len(Y_valid)}\t')

    # make Low dataset.
    datas_l, targets_l = getFromnpz(src, lows, out=True, cla=label)
    datas_l = make_grid(datas_l)
    print(f'testset for measuring OOD performance| Highs: {len(Y_test)}, Lows: {len(targets_l)}')

    ## save
    os.makedirs(save_folder, exist_ok=True)
    names = f'{split(src)[-1]}_{label}' # seg_DE_v

    save_dataset(save_folder, names, 'train', X_train, Y_train)
    save_dataset(save_folder, names, 'valid', X_valid, Y_valid)
    save_dataset(save_folder, names, 'test', X_test, Y_test)

    save_dataset(save_folder, names, 'lows', datas_l, targets_l)

# -----------------------------------------main---------------------------------------------------
# preprocessed data folder location (After 0.0 preprocessing.py)
DATAS = join("C:\\", "Users", "LAPTOP", "jupydir", "DATAS", 'GAMEEMO_npz')
DATA = join(DATAS, 'seg_DE') # segmentation, DE
LABEL = 'v' # 4, v, a
SUBLIST = [str(i).zfill(2) for i in range(1, 29)] # '01', '02', '03', ..., '28'
# ---------------------------------------save data------------------------------------------------
# Sub Independent
# make_dataset_SI(DATA, SUBLIST, LABEL, join(DATAS, 'ID'))
# Sub dependent
# make_dataset_SD(DATA, SUBLIST, LABEL, join(DATAS, 'SubDepen'))


# After 0.2 subdepend.py
folder_name = 'Highs'
## subdepend results
vRANKS = [4,18,24,9,10,1,3,12,8,20,17,6,11,5,7,13,27,23,2,16,19,15,22,21,25,26,28,14]
aRANKS = [4,24,18,10,12,15,9,11,17,3,22,16,5,6,2,7,21,13,8,19,20,1,25,23,27,26,28,14]
make_dataset_HL(DATA, vRANKS, cut=28-6, label='v', save_folder=join(DATAS, folder_name))
make_dataset_HL(DATA, aRANKS, cut=28-5, label='a', save_folder=join(DATAS, folder_name))

# -----------------------------------------check---------------------------------------------------
# load train, valid, test
# def getDataset(path, names, mode):
#     path = join(path, f'{names}_{mode}.npz')
#     data = np.load(path, allow_pickle=True)
#     datas, targets = data['X'], data['Y']
#     return datas, targets
#
# # save_folder = join(DATAS, 'baseline')
# save_folder = join(DATAS, 'SubDepen', '01')
#
# names = f'{split(DATA)[-1]}_{LABEL}'  # seg_DE_v
# X_train, Y_train = getDataset(save_folder , names, 'train')
# X_valid, Y_valid = getDataset(save_folder , names, 'valid')
# X_test, Y_test = getDataset(save_folder , names, 'test')
# print('check')
# print(f'num of train: {len(Y_train)} \t num of valid: {len(Y_valid)} \t num of test: {len(Y_test)}')
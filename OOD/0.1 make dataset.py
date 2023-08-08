import os
from os.path import join
import random
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from utils.constant import *

# -----------------------------------------Setting---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", default="/mnt/data/research_EG", help='After 0.0 preprocessing.py') 
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO", help='GAMEEMO, SEED, SEED_IV, DEAP')
parser.add_argument("--label", dest="label", action="store", default="v", help='v, a :GAMEEMO/DEAP')
args = parser.parse_args()

DATASETS = args.datasets
DATASET_NAME = args.dataset
LABEL = args.label

if DATASET_NAME == 'GAMEEMO':
    DATAS = join(DATASETS, 'GAMEEMO_npz')
    SUB_NUM = GAMEEMO_SUBNUM
elif DATASET_NAME == 'SEED':
    DATAS = join(os.getcwd(),"datasets", "SEED", "npz")
    SUB_NUM = 15
elif DATASET_NAME == 'SEED_IV':
    DATAS = join(os.getcwd(),"datasets", "SEED_IV", "npz")
    SUB_NUM = 15
elif DATASET_NAME == 'DEAP':
    DATAS = join(os.getcwd(),"datasets", "DEAP", "npz")
    SUB_NUM = 32
else:
    print("Unknown Dataset")
    exit(1)


def seed(s):
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
SEED = 42
seed(SEED)


def getFromnpz_(dir, sub, cla='v'):
    sub += '.npz'
    print(sub)
    data = np.load(join(dir, sub), allow_pickle=True)
    datas = data['x']
    if cla == '4': targets = data['y']
    if cla == 'v': targets = data['v']
    if cla == 'a': targets = data['a']
    return datas, targets

def make_dataset(src, sublists, label, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    train_dir, test_dir = join(save_dir, 'train'), join(save_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for sub in sublists:
        datas, targets = getFromnpz_(src, sub, cla=label)
        labels, countsl = np.unique(targets[:, 0], return_counts=True)
        print(f'label {label} count {labels} \t {countsl}')  # labels

        # Make Dataset  ## train 90 : test 10
        X_train, X_test, Y_train, Y_test = train_test_split(datas, targets, test_size=0.1, stratify=targets, random_state=SEED)
        print(f'num of train: {len(Y_train)} \t num of test: {len(Y_test)}\n')

        # save train, test
        np.savez(join(train_dir, f'{label}_{sub}'), X=X_train, Y=Y_train)
        np.savez(join(test_dir, f'{label}_{sub}'), X=X_test, Y=Y_test)
    print(f'saved in {save_dir}')



# -----------------------------------------main---------------------------------------------------
SUBLIST = [str(i).zfill(2) for i in range(1, SUB_NUM + 1)] # '01', '02', '03', ...

make_dataset(join(DATAS,'Preprocessed', 'seg'),   SUBLIST,LABEL, join(DATAS, 'Projects', 'raw'))
make_dataset(join(DATAS,'Preprocessed','seg_DE'), SUBLIST,LABEL, join(DATAS, 'Projects', 'DE'))
make_dataset(join(DATAS,'Preprocessed','seg_PSD'),SUBLIST,LABEL, join(DATAS, 'Projects', 'PSD'))
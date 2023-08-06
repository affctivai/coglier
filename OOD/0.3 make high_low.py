import os
from os.path import join, split
import random
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

from utils.tools import getFromnpz, getFromnpz_
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils.transform import make_grid
from sklearn.model_selection import train_test_split
from utils.constant import *

# -----------------------------------------Setting---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO") # GAMEEMO, SEED, SEED_IV, DEAP
parser.add_argument("--model", dest="model", action="store", default="GAMEEMO") # GAMEEMO, SEED, SEED_IV, DEAP
parser.add_argument("--feature", dest="feature", action="store", default="DE") # DE, PSD
parser.add_argument("--column", dest="column", action="store", default="test_acc") # 기준 칼럼, test_acc, test_loss, roc_auc_score
parser.add_argument("--cut", type= int, dest="cut", action="store", default="4") # low group count
parser.add_argument("--rank", dest="rank", action="store_true") # Rank만 출력




args = parser.parse_args()

DATASET_NAME = args.dataset
MODEL = args.model
FEATURE = args.feature
PROJECT = 'subdepend'
COLUMN = args.column
CUT = args.cut
RANK = args.rank

if DATASET_NAME == 'GAMEEMO':
    DATAS = join("C:\\", "Users", "LAPTOP", "jupydir", "DATAS", 'GAMEEMO_npz')
    SUB_NUM = 28
    CHLS = GAMEEMO_CHLS
    LOCATION = GAMEEMO_LOCATION
    LABEL = 'a' # 4, v, a
elif DATASET_NAME == 'SEED':
    DATAS = join(os.getcwd(),"datasets", "SEED", "npz")
    SUB_NUM = 15
    CHLS = SEED_CHLS
    LOCATION = SEED_LOCATION
    LABEL = '4' # 4, v, a
elif DATASET_NAME == 'SEED_IV':
    DATAS = join(os.getcwd(),"datasets", "SEED_IV", "npz")
    SUB_NUM = 15
    CHLS = SEED_IV_CHLS
    LOCATION = SEED_IV_LOCATION
    LABEL = '4' # 4, v, a
elif DATASET_NAME == 'DEAP':
    DATAS = join(os.getcwd(),"datasets", "DEAP", "npz")
    SUB_NUM = 32
    CHLS = DEAP_CHLS
    LOCATION = DEAP_LOCATION
    LABEL = 'v' # 4, v, a
else:
    print("Unknown Dataset")
    exit(1)

def set_args(project, model_name, feature, label): # 0.1 make dataset과 호환맞춘다면 편의대로...
    if model_name == 'CCNN':
        project_data = '_'.join([project, feature, 'grid'])
        project_name = '_'.join([project, model_name, feature])

    elif model_name in ['TSC', 'EEGNet']:
        project_data = '_'.join([project, 'raw'])
        project_name = '_'.join([project, model_name])

    elif model_name == 'DGCNN':
        project_data = '_'.join([project, feature])
        project_name = '_'.join([project, model_name, feature])

    if label == 'a':    train_name = 'arousal'
    elif label == 'v':  train_name = 'valence'
    else:               train_name = 'emotion'

    data_dir = join(DATAS, project_data)
    data_name = f'{LABEL}'
    return data_dir, data_name, project_name, train_name

DATA, NAME, project_name, train_name = set_args(PROJECT, MODEL, FEATURE, LABEL)

def seed(s):
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
SEED = 42
seed(SEED)


def save_dataset(folder, file_name, x, y, out=True):
    np.savez(join(folder, file_name), X=x, Y=y)
    if out: print(f'saved in {folder}')

def scaling(datas, scaler_name = None):
    if scaler_name == None: return datas
    flattend = datas.reshape(-1, 1)

    if scaler_name == 'standard':
        scaler = StandardScaler()
        scaled_datas = scaler.fit_transform(flattend)

    if scaler_name == 'robust':
        scaler = RobustScaler()
        scaled_datas = scaler.fit_transform(flattend)

    if scaler_name == 'log':
        scaled_datas = np.log1p(datas)

    if scaler_name == 'log_standard':
        scaler = StandardScaler()
        scaled_datas = scaler.fit_transform(np.log1p(flattend))

    scaled_datas = scaled_datas.reshape(datas.shape)
    return scaled_datas

def deshape(datas, shape_name = None):
    if shape_name == None: return datas

    # for CCNN model (samples, channels, 4 bands) -> (samples, 4 bands, 9, 9)
    if shape_name == 'grid':
        datas = make_grid(datas, CHLS, LOCATION)
        print(f'grid (samples, 4freq, 9x9): {datas.shape}')

    # for TSCeption, EEGnet (samples, channels, window) -> (samples, 1, channels, window)
    if shape_name == 'expand':
        datas = np.expand_dims(datas, axis=1)
        print(f'expand (samples, 1, channels, window): {datas.shape}')
    return datas

# Subject Independent
def make_dataset_SI(src, sublists, label, scaler_name, shape_name, save_folder):
    datas, targets = getFromnpz(src, sublists, out=True, cla=label)
    labels, countsl = np.unique(targets[:, 0], return_counts=True)
    subIDs, countss = np.unique(targets[:, 1], return_counts=True)

    print(f'data shape: {datas.shape} target shape: {targets.shape}')
    print(f'label {label} count {labels} \t {countsl}') # labels
    print(f'subID count {subIDs} \t {countss}\n') # subIDs

    # transform
    datas = scaling(datas, scaler_name)
    datas = deshape(datas, shape_name)

    # Make Dataset  ## train 80 : valid 10 : test 10
    X_train, X, Y_train, Y = train_test_split(datas, targets, test_size=0.2, stratify=targets, random_state=SEED)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=SEED)
    print(f'num of train: {len(Y_train)} \t num of valid: {len(Y_valid)} \t num of test: {len(Y_test)}\n')

    ## save train, valid, test
    os.makedirs(save_folder, exist_ok=True)

    save_dataset(save_folder, f'{label}_train', X_train, Y_train)
    save_dataset(save_folder, f'{label}_valid', X_valid, Y_valid)
    save_dataset(save_folder, f'{label}_test', X_test, Y_test)

# Subject dependent
def make_dataset_SD(src, sublists, label, scaler_name, shape_name, save_folder):
    for sub in sublists:
        datas, targets = getFromnpz_(src, sub, out=True, cla=label)
        labels, countsl = np.unique(targets[:, 0], return_counts=True)
        print(f'label {label} count {labels} \t {countsl}')  # labels

        # transform
        datas = scaling(datas, scaler_name)
        datas = deshape(datas, shape_name)

        # Make Dataset  ## train 80 : valid 10 : test 10
        X_train, X, Y_train, Y = train_test_split(datas, targets, test_size=0.2, stratify=targets, random_state=SEED)
        X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=SEED)

        ## save train, valid, test
        save_folder_sub = join(save_folder, sub)
        os.makedirs(save_folder_sub, exist_ok=True)

        save_dataset(save_folder_sub, f'{label}_train', X_train, Y_train, out=False)
        save_dataset(save_folder_sub, f'{label}_valid', X_valid, Y_valid, out=False)
        save_dataset(save_folder_sub, f'{label}_test', X_test, Y_test, out=False)

    print(f'num of train: {len(Y_train)} \t num of valid: {len(Y_valid)} \t num of test: {len(Y_test)}\n')

# High Low dataset
def make_dataset_HL(src, ranks, cut, label, scaler_name, shape_name, save_folder):
    ranks = [str(sub).zfill(2) for sub in ranks]
    # cut = int(28 * cut_rate)
    higs = ranks[: cut]
    lows = ranks[cut :]
    print(label, 'high', len(higs), '명', higs)
    print(label, 'low ', len(lows), '명', lows)

    # make High dataset. train 80 : valid 10 : test 10
    datas_h, targets_h = getFromnpz(src, higs, out=True, cla=label)

    # transform
    datas_h = scaling(datas_h, scaler_name)
    datas_h = deshape(datas_h, shape_name)

    X_train, X, Y_train, Y = train_test_split(datas_h, targets_h, test_size=0.2, stratify=targets_h, random_state=SEED)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=SEED)
    print(f'High train: {len(Y_train)} \t High valid: {len(Y_valid)}\t')

    # make Low dataset.
    datas_l, targets_l = getFromnpz(src, lows, out=True, cla=label)
    datas_l = make_grid(datas_l, CHLS, LOCATION)
    print(f'testset for measuring OOD performance| Highs: {len(Y_test)}, Lows: {len(targets_l)}')

    ## save
    os.makedirs(save_folder, exist_ok=True)

    save_dataset(save_folder, f'{label}_train', X_train, Y_train)
    save_dataset(save_folder, f'{label}_valid', X_valid, Y_valid)
    save_dataset(save_folder, f'{label}_test', X_test, Y_test)

    save_dataset(save_folder, f'{label}_lows', datas_l, targets_l)


## subdepend results
# vRANKS = [4,18,24,9,10,1,3,12,8,20,17,6,11,5,7,13,27,23,2,16,19,15,22,21,25,26,28,14]
# aRANKS = [4,24,18,10,12,15,9,11,17,3,22,16,5,6,2,7,21,13,8,19,20,1,25,23,27,26,28,14]



project_path = train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, project_name))
print('Read result from: ', project_path)

result = pd.read_excel(join(project_path, f'{train_name}_results.xlsx'))
col = result[COLUMN].to_numpy()
rank = np.argsort(col)[::-1] + 1
col = np.sort(col)[::-1]
print('SUB ID: ', rank)
print(f'{COLUMN}:', col)

if not RANK:
    if MODEL == 'CCNN':
        make_dataset_HL(join(DATAS,'Preprocessed','seg_DE'),rank, SUB_NUM-CUT,LABEL,None,'grid', join(DATAS, 'Projects', f'High_DE_grid_{CUT}'))
        make_dataset_HL(join(DATAS,'Preprocessed','seg_PSD'),rank, SUB_NUM-CUT,LABEL,'log','grid', join(DATAS, 'Projects', f'High_PSD_grid_{CUT}'))
    elif MODEL == 'TSC' or MODEL == 'EEGNet':
        make_dataset_HL(join(DATAS,'Preprocessed','seg'),rank, SUB_NUM-CUT,LABEL,'standard','expand', join(DATAS, 'Projects', f'High_raw_{CUT}'))
    elif MODEL == 'DGCNN':
        make_dataset_HL(join(DATAS,'Preprocessed','seg_DE'),rank,SUB_NUM-CUT,LABEL,None,None, join(DATAS, 'Projects', f'High_DE_{CUT}'))
        make_dataset_HL(join(DATAS,'Preprocessed','seg_PSD'),rank,SUBNUM_CUT,LABEL,'log',None, join(DATAS, 'Projects', f'High_PSD_{CUT}'))

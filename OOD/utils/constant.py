import os
from os.path import join

GAMEEMO_CHLS = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8']  # 14 channels
GAMEEMO_LOCATION = [['-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
    ['F7', '-', 'F3', '-', '-', '-', 'F4', '-', 'F8'],
    ['-', 'FC5', '-', '-', '-', '-', '-', 'FC6', '-'],
    ['T7', '-', '-', '-', '-', '-', '-', '-', 'T8'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['P7', '-', '-', '-', '-', '-', '-', '-', 'P8'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', 'O1', '-', 'O2', '-', '-', '-']]
GAMEEMO_SUBNUM = 28


DEAP_CHLS = [
    'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3',
    'P7', 'PO3', 'O1', 'OZ', 'PZ', 'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2',
    'CZ', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]

DEAP_LOCATION = [['-', '-', '-', 'FP1', '-', 'FP2', '-', '-', '-'],
                      ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
                      ['F7', '-', 'F3', '-', 'FZ', '-', 'F4', '-', 'F8'],
                      ['-', 'FC5', '-', 'FC1', '-', 'FC2', '-', 'FC6', '-'],
                      ['T7', '-', 'C3', '-', 'CZ', '-', 'C4', '-', 'T8'],
                      ['-', 'CP5', '-', 'CP1', '-', 'CP2', '-', 'CP6', '-'],
                      ['P7', '-', 'P3', '-', 'PZ', '-', 'P4', '-', 'P8'],
                      ['-', '-', '-', 'PO3', '-', 'PO4', '-', '-', '-'],
                      ['-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-']]
DEAP_SUBNUM = 32


SEED_CHLS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

SEED_LOCATION = [
    ['-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-'],
    ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-'],
    ['-', '-', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '-', '-']
]
SEED_SUBNUM = 15


SEED_IV_CHLS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

SEED_IV_LOCATION = [
    ['-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-'],
    ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-'],
    ['-', '-', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '-', '-']
]
SEED_IV_SUBNUM = 15

def load_dataset_info(dataset):
    if dataset == 'GAMEEMO':
        DATAS = join(DATASETS, 'GAMEEMO_npz', 'Projects')
        SUB_NUM = GAMEEMO_SUBNUM
        CHLS = GAMEEMO_CHLS
        LOCATION = GAMEEMO_LOCATION
        return DATAS, SUB_NUM, CHLS, LOCATION
    elif dataset == 'SEED':
        DATAS = join(os.getcwd(),"datasets", dataset, "npz", "Projects")
        SUB_NUM = SEED_SUBNUM
        CHLS = SEED_CHLS
        LOCATION = SEED_LOCATION
        return DATAS, SUB_NUM, CHLS, LOCATION
    elif dataset == 'SEED_IV':
        DATAS = join(os.getcwd(),"datasets", dataset, "npz", "Projects")
        SUB_NUM = SEED_IV_SUBNUM
        CHLS = SEED_IV_CHLS
        LOCATION = SEED_IV_LOCATION
        return DATAS, SUB_NUM, CHLS, LOCATION
    elif dataset == 'DEAP':
        DATAS = join(os.getcwd(),"datasets", dataset, "npz", "Projects")
        SUB_NUM = DEAP_SUBNUM
        CHLS = DEAP_CHLS
        LOCATION = DEAP_LOCATION
        return DATAS, SUB_NUM, CHLS, LOCATION
    else:
        print("Unknown Dataset")
        exit(1)
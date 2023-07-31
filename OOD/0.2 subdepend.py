import os
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", dest="model", action="store", default="CCNN") # CCNN, TSC, EEGNet, DGCNN
parser.add_argument("--label", dest="label", action="store", default="v") # 4, v, a
parser.add_argument("--batch", dest="batch", action="store", default="64") # 64, 128
parser.add_argument("--feature", dest="feature", action="store", default="DE") # DE, PSD
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO") # GAMEEMO, SEED, SEED_IV, DEAP
parser.add_argument("--epoch", dest="epoch", action="store", default="1") # 1, 50, 100
parser.add_argument("--test", dest="test", action="store_true")

args = parser.parse_args()

DATASET_NAME = args.dataset
LABEL = args.label
MODEL_NAME = args.model
FEATURE = args.feature
BATCH = int(args.batch)
EPOCH = int(args.epoch)
FEATURE = args.feature
TEST = args.test
PROJECT = 'subdepend'

if DATASET_NAME == 'GAMEEMO':
    DATAS = join("C:\\", "Users", "LAPTOP", "jupydir", "DATAS", 'GAMEEMO_npz', 'Projects')
    # LABEL = 'v'     # 4, v, a
    # PROJECT = 'baseline'
    # MODEL_NAME = 'DGCNN'    # 'CCNN', 'TSC', 'EEGNet', 'DGCNN'
    # FEATURE = 'PSD'          # 'DE', 'PSD'
    # BATCH = 64
    # SUBNUMS = 
elif DATASET_NAME == 'SEED':
    DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")
    SUBNUMS = 15
    # LABEL = '4' # 4, v, a
    # EPOCH = 1
    # BATCH = 128
elif DATASET_NAME == 'SEED_IV':
    DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")
    SUBNUMS = 15
    # LABEL = '4' # 4, v, a
    # EPOCH = 100
    # BATCH = 128
elif DATASET_NAME == 'DEAP':
    DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")
    SUBNUMS = 32
    # LABEL = 'v' # 4, v, a
    # EPOCH = 1
    # BATCH = 64
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

DATA, NAME, project_name, train_name = set_args(PROJECT, MODEL_NAME, FEATURE, LABEL)


def run(sublist):
    for sub in sublist:
        print(sub)
        subprocess.run(f'{sys.executable} subdepend.py --subID={sub} --batch={BATCH} --epoch={EPOCH} --target={LABEL} --project_name={PROJECT} --feature={FEATURE} --dataset={DATASET_NAME} --model={MODEL_NAME}', shell=True)
def save_results(sublist):
    test_results = dict()
    project_path = train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, project_name))
    for sub in sublist:
        file = open(join(project_path, sub, train_name, 'test.txt'), 'r')
        result = '{'+ file.readline() + '}'
        test_results[int(sub)] = eval(result)

    df = pd.DataFrame.from_dict(test_results, orient='index')
    df.to_excel(join(project_path, f'{train_name}_results.xlsx'))

SUBLIST = [str(i).zfill(2) for i in range(1, SUBNUMS+1)]

if not TEST:
    run(SUBLIST)
save_results(SUBLIST)
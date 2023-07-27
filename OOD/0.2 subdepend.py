import os
from os.path import join
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
parser.add_argument("--feature", dest="feature", action="store", default="DE") # DE, PSD
parser.add_argument('--target', type=str, default = 'v') # 4, v, a

args = parser.parse_args()

DATASET_NAME = args.dataset
LABEL = args.label
MODEL_NAME = args.model
FEATURE = args.feature
BATCH = int(args.batch)
EPOCH = int(args.epoch)
FEATURE = args.feature
LABEL = args.target

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


def run(sublist):
    for sub in sublist:
        print(sub)
        subprocess.run(f'{sys.executable} subdepend.py --subID={sub} --batch={BATCH} --epoch={EPOCH} --target={LABEL} --project_name={project_name} --feature={FEATURE} --target={LABEL} --dataset={DATASET_NAME}')

def save_results(sublist):
    test_results = dict()
    for sub in sublist:
        file = open(join(projcet_path, sub, train_name, 'test.txt'), 'r')
        result = '{'+ file.readline() + '}'
        test_results[int(sub)] = eval(result)

    df = pd.DataFrame.from_dict(test_results, orient='index')
    df.to_excel(join(projcet_path, f'{train_name}_results.xlsx'))

project_name = 'subdepend'
projcet_path = join(os.getcwd(), 'results', DATASET_NAME, project_name)

lb = '4'
if lb == 'a': train_name = 'arousal'
elif lb == 'v': train_name = 'valence'
else: train_name = 'emotion'

SUBLIST = [str(i).zfill(2) for i in range(1, SUBNUMS+1)]

run(SUBLIST)
save_results(SUBLIST)
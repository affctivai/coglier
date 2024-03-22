import os
from os.path import join
from pathlib import Path
import pandas as pd
import subprocess
import sys
import argparse

from utils.constant import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO", help='GAMEEMO, SEED, SEED_IV')
parser.add_argument("--label", type=str, default='v', help='v, a :GAMEEMO')
parser.add_argument("--model", dest="model", action="store", default="CCNN", help='CCNN, TSC, DGCNN')
parser.add_argument("--feature", dest="feature", action="store", default="DE", help='DE, PSD, raw')
parser.add_argument('--batch', type=int, default = 64)
parser.add_argument('--epoch', type=int, default = 3)
parser.add_argument('--project_name', type=str, default = 'Subdepend')
parser.add_argument("--test", dest="test", action="store_true")
args = parser.parse_args()

DATASET_NAME = args.dataset
LABEL = args.label
MODEL_NAME = args.model
FEATURE = args.feature
BATCH = args.batch
EPOCH = args.epoch
PROJECT = args.project_name
TEST = args.test
args = parser.parse_args()

DATAS, SUB_NUM, CHLS, LOCATION = load_dataset_info(DATASET_NAME)

if LABEL == 'a':    train_name = 'arousal'
elif LABEL == 'v':  train_name = 'valence'
else:               train_name = 'emotion'

SUBLIST = [str(i).zfill(2) for i in range(1, SUB_NUM+1)]

def run(sublist):
    for sub in sublist:
        print(sub)
        subprocess.run(f'{sys.executable} subdepend.py --dataset={DATASET_NAME} --subID={sub} '
                       f'--label={LABEL} --model={MODEL_NAME} --feature={FEATURE} --batch={BATCH} --epoch={EPOCH} '
                       f'--project_name={PROJECT}', shell=True)

def save_results(sublist):
    test_results = dict()
    if MODEL_NAME == 'TSC': MODEL_FEATURE = MODEL_NAME
    else: MODEL_FEATURE = '_'.join([MODEL_NAME, FEATURE])
    project_path = train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, MODEL_FEATURE, PROJECT))
    for sub in sublist:
        file = open(join(project_path, sub, train_name, 'test.txt'), 'r')
        result = '{'+ file.readline() + '}'
        test_results[int(sub)] = eval(result)

    df = pd.DataFrame.from_dict(test_results, orient='index')
    df.to_excel(join(project_path, f'{train_name}_results.xlsx'))

if not TEST:
    run(SUBLIST)
save_results(SUBLIST)
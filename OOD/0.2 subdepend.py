import os
from os.path import join
import pandas as pd
import numpy as np
import subprocess
import sys

# ---- GAMEEMO
# DATASET_NAME = "GAMEEMO"
# # LABEL = 'v' # 4, v, a
# EPOCH = 200
# BATCH = 64
# SUBNUMS = 


# ---- DEAP
# DATASET_NAME = "DEAP"
# LABEL = 'v' # 4, v, a
# EPOCH = 1
# BATCH = 64
# SUBNUMS = 32

# ---- SEED_IV
DATASET_NAME = "SEED_IV"
LABEL = '4' # 4, v, a
EPOCH = 1
BATCH = 128
SUBNUMS = 1

# ---- SEED
# DATASET_NAME = "SEED"
# LABEL = '4' # 4, v, a
# EPOCH = 1
# BATCH = 128
# SUBNUMS = 15

def run(sublist):
    for sub in sublist:
        print(sub)
        subprocess.run(f'{sys.executable} subdepend.py --subID={sub} --batch={BATCH} --epoch={EPOCH} --target={LABEL} --project_name={project_name}')

def save_results(sublist):
    test_results = dict()
    for sub in sublist:
        file = open(join(projcet_path, sub, train_name, 'test.txt'), 'r')
        result = '{'+ file.readline() + '}'
        test_results[int(sub)] = eval(result)

    df = pd.DataFrame.from_dict(test_results, orient='index')
    df.to_excel(join(projcet_path, f'{train_name}_results.xlsx'))

project_name = 'Subdepend_de'
# project_name = 'Subdepend_EEGNet'
# project_name = 'Subdepend_TSC'
projcet_path = join(os.getcwd(), 'results', DATASET_NAME, project_name)

lb = '4'
if lb == 'a': train_name = 'arousal'
elif lb == 'v': train_name = 'valence'
else: train_name = 'emotion'

SUBLIST = [str(i).zfill(2) for i in range(1, SUBNUMS+1)]

run(SUBLIST)
save_results(SUBLIST)
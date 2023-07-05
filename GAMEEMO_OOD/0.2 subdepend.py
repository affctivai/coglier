import os
from os.path import join
import pandas as pd
import numpy as np
import subprocess

def run(sublist):
    for sub in sublist:
        print(sub)
        subprocess.run(f'python subdepend.py --subID={sub} --batch={BATCH} --epoch={EPOCH} --target={LABEL}\
         --project_name={project_name} --dname={DNAME}')

def save_results(sublist):
    test_results = dict()
    for sub in sublist:
        file = open(join(projcet_path, sub, train_name, 'test.txt'), 'r')
        result = '{'+ file.readline() + '}'
        test_results[int(sub)] = eval(result)

    df = pd.DataFrame.from_dict(test_results, orient='index')
    df.to_excel(join(projcet_path, f'{train_name}_results.xlsx'))

# main-----------------------------------------------------------------------------------------
project_name = 'subdepend_de'
# project_name = 'subdepend_psd'
projcet_path = join(os.getcwd(), 'results', project_name+'_CCNN')
DNAME = 'seg_DE'
# DNAME = 'seg_PSD'
LABEL = 'v'
if LABEL == 'a': train_name = 'arousal'
elif LABEL == 'v': train_name = 'valence'
SUBLIST = [str(i).zfill(2) for i in range(1, 29)] # '01', '02', '03', ..., '28'
BATCH = 64
EPOCH = 120

# run(SUBLIST)
save_results(SUBLIST)
# run(['26'])

## run(['28']) # If there are subjects with errors, run again.
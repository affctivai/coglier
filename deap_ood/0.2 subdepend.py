import os
from os.path import join
import pandas as pd
import numpy as np
import subprocess
import sys

def run(sublist):
    for sub in sublist:
        print(sub)
        subprocess.run(f'{sys.executable} subdepend.py --subID={sub} --batch={BATCH} --epoch={EPOCH} --target={lb} --project_name={project_name}')

def save_results(sublist):
    test_results = dict()
    for sub in sublist:
        file = open(join(projcet_path, sub, train_name, 'test.txt'), 'r')
        result = '{'+ file.readline() + '}'
        test_results[int(sub)] = eval(result)

    df = pd.DataFrame.from_dict(test_results, orient='index')
    df.to_excel(join(projcet_path, f'{train_name}_results.xlsx'))

project_name = 'Subdepend'
projcet_path = join(os.getcwd(), 'results', project_name)

lb = 'v'
if lb == 'a': train_name = 'arousal'
elif lb == 'v': train_name = 'valence'

SUBLIST = [str(i).zfill(2) for i in range(1, 33)]

BATCH = 64
EPOCH = 120

run(SUBLIST)
save_results(SUBLIST)
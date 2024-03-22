import os
from os.path import join
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from utils.constant import *
from utils.tools import getFromnpz_, seed_everything

from pyriemann.estimation import Covariances
from pyriemann.clustering import Potato

# -----------------------------------------Setting---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO", help='GAMEEMO, SEED, SEED_IV')
parser.add_argument("--label", dest="label", action="store", default="v", help='v, a :GAMEEMO')
parser.add_argument("--z", dest="z", type=float, action="store", default=2.5, help='z threshold for riemannian potato')
args = parser.parse_args()

DATASET_NAME = args.dataset
LABEL = args.label
Z_THRESHOLD = args.z

if DATASET_NAME == 'GAMEEMO':
    DATAS = join(os.getcwd(),"datasets", "GAMEEMO", "npz")
    SUB_NUM = GAMEEMO_SUBNUM
elif DATASET_NAME == 'SEED':
    DATAS = join(os.getcwd(),"datasets", "SEED", "npz")
    SUB_NUM = SEED_SUBNUM
elif DATASET_NAME == 'SEED_IV':
    DATAS = join(os.getcwd(),"datasets", "SEED_IV", "npz")
    SUB_NUM = SEED_IV_SUBNUM
else:
    print("Unknown Dataset")
    exit(1)

random_seed = 42
seed_everything(random_seed)

def make_dataset(sublists, label):
    cov_trains, cov_tests, index_trains, index_tests = dict(), dict(), dict(), dict()
    rp_train = []
    for sub in sublists:
        raws, targets = getFromnpz_(join(DATAS,'Preprocessed', 'seg'), sub, cla=label)

        index = np.arange(len(targets))
        # Make Dataset  ## train 90 : test 10
        index_train, index_test, Y_train, Y_test = train_test_split(index, targets, test_size=0.1, stratify=targets, random_state=random_seed)
        if len(Y_train) >= 5000:
            rp_index, _, y_rp_train, _ = train_test_split(index_train, Y_train, test_size=0.8, stratify=Y_train, random_state=random_seed)
            print(f'num of rp train: {len(y_rp_train)} \t num of train: {len(Y_train)} \t num of test: {len(Y_test)}\n')
        else:
            print(f'num of train: {len(Y_train)} \t num of test: {len(Y_test)}\n')
            

        # fit pyriemann
        raws = raws.astype('float64')
        covs = Covariances(estimator='lwf').transform(raws)
        if len(Y_train) >= 5000:
            cov_rp = covs[rp_index]
        else:
            cov_rp = covs[index_train]
        cov_train = covs[index_train]
        cov_test = covs[index_test]
        
        rp_train.extend(cov_rp)
        cov_trains[sub] = cov_train
        cov_tests[sub] = cov_test
        index_trains[sub] = index_train
        index_tests[sub] = index_test
    
    rp_train = np.array(rp_train)
    print(f'Fit riemannian potato size: {len(rp_train)}')

    rp = Potato(metric='riemann', threshold=Z_THRESHOLD)
    rp.fit(rp_train)
    print("Fit riemannian potato finished")

    for sub in sublists:
        cov_train = cov_trains[sub]
        cov_test = cov_tests[sub]
        index_train = index_trains[sub]
        index_test = index_tests[sub]

        # predict with pyriemman & split index
        pr_train = rp.predict(cov_train)
        pr_test = rp.predict(cov_test)
        print(f'prediction in train: {np.unique(pr_train, return_counts=True)}')
        print(f'prediction in test: {np.unique(pr_test, return_counts=True)}')

        index_train = index_train[pr_train == 1]
        index_test = index_test[pr_test == 1]
        # save train, test
        for pre_src, pro_src in [('seg', 'raw'), ('seg_DE','DE'), ('seg_PSD', 'PSD')]:
            src_dir = join(DATAS,'Preprocessed', pre_src)
            save_dir = join(DATAS,'Projects', pro_src)
            os.makedirs(save_dir, exist_ok=True)
            train_dir, test_dir = join(save_dir, 'train'), join(save_dir, 'test')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            datas, targets = getFromnpz_(src_dir, sub, cla=label)
            labels, countsl = np.unique(targets[:, 0], return_counts=True)
            X_train = datas[index_train]
            Y_train = targets[index_train]
            X_test = datas[index_test]
            Y_test = targets[index_test]
            print(f'f{pro_src} total data: {len(datas)}, train data: {len(X_train)}, test data: {len(X_test)}')

            np.savez(join(train_dir, f'{label}_{sub}_rp_{int(Z_THRESHOLD*100)}'), X=X_train, Y=Y_train)
            np.savez(join(test_dir, f'{label}_{sub}_rp_{int(Z_THRESHOLD*100)}'), X=X_test, Y=Y_test)

# -----------------------------------------main---------------------------------------------------
SUBLIST = [str(i).zfill(2) for i in range(1, SUB_NUM + 1)] # '01', '02', '03', ...

make_dataset(SUBLIST,LABEL)
import os
from os.path import join, exists
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils.constant import *
from utils.transform import scaling, deshape
from utils.dataset import load_list_subjects, PreprocessedDataset
from utils.model import get_model

from utils.tools import get_roc_auc_score, print_auroc
from utils.tools import seed_everything, get_folder

random_seed = 42
seed_everything(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO", help='GAMEEMO, SEED, SEED_IV, DEAP')
parser.add_argument("--label", dest="label", action="store", default="v", help='v, a :GAMEEMO/DEAP')
parser.add_argument("--model", dest="model", action="store", default="CCNN", help='CCNN, TSC, EEGNet, DGCNN')
parser.add_argument("--feature", dest="feature", action="store", default="DE", help='DE, PSD, raw')
parser.add_argument("--batch", dest="batch", type=int, action="store", default=64) # 64, 128

parser.add_argument("--column", dest="column", action="store", default="test_acc", help='test_acc, test_loss, roc_auc_score') # 기준 칼럼
parser.add_argument("--cut", type= int, dest="cut", action="store", default="4") # low group count
parser.add_argument("--thresholds", type=str, dest="thresholds", action="store", default='0.80 0.85 0.90 0.95')
args = parser.parse_args()

DATASET_NAME = args.dataset

if DATASET_NAME == 'SEED': label_names = SEED_LABELS 
elif DATASET_NAME=='SEED_IV': label_names = SEED_IV_LABELS 

LABEL = args.label
MODEL_NAME = args.model
FEATURE = args.feature
BATCH = args.batch

COLUMN = args.column
CUT = args.cut
THRESHOLDS = list(map(float, args.thresholds.split()))

PROJECT = f'High' or f'Low_{CUT}'

if MODEL_NAME == 'CCNN': SHAPE = 'grid'
elif MODEL_NAME == 'TSC' or MODEL_NAME == 'EEGNet': SHAPE = 'expand'; FEATURE = 'raw'
elif MODEL_NAME == 'DGCNN': SHAPE = None
if FEATURE == 'DE': SCALE = None
elif FEATURE == 'PSD': SCALE = 'log'
elif FEATURE == 'raw': SCALE = 'standard'
if LABEL == 'a':    train_name = 'arousal'
elif LABEL == 'v':  train_name = 'valence'
else:               train_name = 'emotion'
if MODEL_NAME == 'EEGNet' or MODEL_NAME == 'TSC': MODEL_FEATURE = MODEL_NAME
else: MODEL_FEATURE = '_'.join([MODEL_NAME, FEATURE])

DATAS, SUB_NUM, CHLS, LOCATION = load_dataset_info(DATASET_NAME)
DATA = join(DATAS, FEATURE)

train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, MODEL_FEATURE, PROJECT, train_name))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_ID(): # After subdepend.py
    subdepend_result_path = Path(join(os.getcwd(), 'results', DATASET_NAME, MODEL_FEATURE, 'Subdepend'))
    print('Read subject-dependent result from: ', subdepend_result_path)
    result = pd.read_excel(join(subdepend_result_path, f'{train_name}_results.xlsx'))
    col = result[COLUMN].to_numpy()
    if COLUMN != 'test_loss':
        rank = np.argsort(col)[::-1] + 1
        col = np.sort(col)[::-1]
    else:
        rank = np.argsort(col) + 1
        col = np.sort(col)
    print('SUB ID: ', rank)
    print(f'{COLUMN}:', col)
    
    ranks = [str(sub).zfill(2) for sub in rank]

    highs = ranks[: SUB_NUM-CUT]
    lows = ranks[SUB_NUM-CUT :]
    return highs, lows


def evaluate_test(model, loader, criterion, device):
    losss, accs,  = [], []
    labels, preds, msps, subIDs = [], [], [], []
    model.eval() 
    with torch.no_grad():
        for (x, y, subID) in loader:
            x = x.to(device);   y = y.to(device)

            y_pred = model(x)

            msp = nn.functional.softmax(y_pred, dim=-1)
            msp, maxidx = msp.max(1)

            loss = criterion(y_pred, y)
            losss.append(loss.cpu())
            accs.append(y.eq(maxidx).cpu())
            labels.append(y.cpu())
            preds.append(maxidx.cpu())
            msps.append(msp.cpu())
            subIDs.append(subID.cpu())
    accs = torch.cat(accs, dim=0)
    losss = torch.cat(losss, dim=0)    
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    msps = torch.cat(msps, dim=0)
    subIDs = torch.cat(subIDs, dim=0)
    return losss, accs, labels, preds, msps, subIDs

def analysis(train_path, save=True):
    if not exists(train_path): raise FileNotFoundError(f"File not found: {train_path}, Set the train weight path properly.")
    
    analysis_path = Path(join(train_path, 'analysis'))
    analysis_path = get_folder(analysis_path)

    SUBLIST = [str(i).zfill(2) for i in range(1, SUB_NUM + 1)]  # '01', '02', ...
    datas, targets = load_list_subjects(DATA, 'train', SUBLIST, LABEL)
    datas = scaling(datas, scaler_name=SCALE)
    datas = deshape(datas, shape_name=SHAPE, chls=CHLS, location=LOCATION)
    dataset = PreprocessedDataset(datas, targets)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
    labels_name = np.unique(dataset.y) + 1
    model, _ = get_model(MODEL_NAME, dataset.x.shape, len(labels_name), device)
    model.load_state_dict(torch.load(join(train_path, 'best.pt')))
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)
    n_samples = len(dataset)

    _, _, _, _, msps, _ = evaluate_test(model, loader, criterion, device)
    if LABEL == 'a': colors = ['darkorange', 'silver'] # OOD ID
    else: colors = ['#4a7fb0', '#adc6e5']

    analysis_path.mkdir(parents=True, exist_ok=True)
    with open(join(analysis_path, 'analysis.txt'), 'w') as file:
        HIGS, LOWS = get_ID()
        file.write(f'HIGS {len(HIGS)} {HIGS}\nLOWS {len(LOWS)} {LOWS}\n')

        for threshold in THRESHOLDS:
            ind_idxs = msps >= float(threshold)
            ood_idxs = msps < float(threshold)
            
            n_ind = ind_idxs.sum().item()
            n_ood = n_samples - n_ind
            ind_rate, ood_rate = n_ind/n_samples*100, n_ood/n_samples*100
            file.write(f'\nT:{threshold}\tID/OOD count|ratio : {n_ind},{n_ood}|{ind_rate:.1f}%,{ood_rate:.1f}%\n')

            datas_ind, targets_ind = datas[ind_idxs], targets[ind_idxs]
            datas_ood, targets_ood = datas[ood_idxs], targets[ood_idxs]

            if save:
                np.savez(join(analysis_path, f'IND_{int(threshold*100)}'), X=datas_ind, Y=targets_ind)
                np.savez(join(analysis_path, f'OOD_{int(threshold*100)}'), X=datas_ood, Y=targets_ood)

            ## plot ID & OOD per subject---------------------------------------------------------------------
            subids, ind_subs = np.unique(targets_ind[:, 1], return_counts=True)
            _, ood_subs = np.unique(targets_ood[:, 1], return_counts=True)
            file.write(f'{subids}\n{ind_subs}\n{ood_subs}\n')

            # Calculate the ratio
            ind_ratios = ind_subs / (ind_subs + ood_subs) * 100
            ood_ratios = ood_subs / (ind_subs + ood_subs) * 100

            # Sort the data
            sorted_indices = sorted(range(len(ood_subs)), key=lambda i: ood_subs[i], reverse=True)

            sorted_subids = [subids[i] for i in sorted_indices]
            sorted_ind_subs = ind_subs[sorted_indices]
            sorted_ood_subs = ood_subs[sorted_indices]
            sorted_ind_ratios = ind_ratios[sorted_indices]
            sorted_ood_ratios = ood_ratios[sorted_indices]

            # Create the bar graph
            width = 0.75  # width of the bars
            ind = range(len(sorted_subids))  # the x locations for the groups

            fig, ax = plt.subplots(figsize=(10, 10))

            ood_bars = ax.bar(ind, sorted_ood_subs, width, label='OOD', color=colors[0])
            ind_bars = ax.bar(ind, sorted_ind_subs, width, label='ID', bottom=sorted_ood_subs, color=colors[1])

            # Add labels, title, and legend
            ax.set_title(f'OOD/ID Count per Subject (T:{threshold} Remove-Rate:{ood_rate:.1f}%) on {DATASET_NAME}-{train_name}', fontsize=16)
            ax.set_xticks(ind)
            ax.set_xticklabels(sorted_subids, fontsize=10)
            ax.set_xlabel('Subject ID', fontsize=13)
            ax.set_ylabel('Sample Count', fontsize=15)
            ax.legend(fontsize=20) #loc='upper left'

            # Annotate bars
            for i, (bar, ro) in enumerate(zip(ood_bars, sorted_ood_ratios)):
                plt.text(bar.get_x()+bar.get_width()/2.0, bar.get_height(), f'{ro:.0f}%', ha='center', va='top')                     

            plt.tight_layout()
            plt.savefig(join(analysis_path, f'ID_OOD_subid{int(threshold*100)}.png'), dpi=300)
            
            ## plot ID & OOD per class----------------------------------------------------------------------------
            class_id, ind_class = np.unique(targets_ind[:, 0], return_counts=True)
            _, ood_class = np.unique(targets_ood[:, 0], return_counts=True)
            file.write(f'{class_id}\n{ind_class}\n{ood_class}\n')

            # Calculate the ratio
            ind_ratios_c = ind_class / (ind_class + ood_class) * 100
            ood_ratios_c = ood_class / (ind_class + ood_class) * 100

            # Create the bar graph
            class_id = class_id + 1
            fig, ax = plt.subplots(figsize=(10, 10))

            ood_bars = ax.bar(class_id, ood_class, width, label='OOD', color=colors[0])
            ind_bars = ax.bar(class_id, ind_class, width, label='ID', bottom=ood_class, color=colors[1])

            ax.set_title(f'OOD/ID Count per Class (T:{threshold} Remove-Rate:{ood_rate:.1f}%) on {DATASET_NAME}', fontsize=18)
            ax.set_xticks(class_id)
            label_names = ['negative', 'neutral', 'positive'] 
            ax.set_xticklabels(label_names, fontsize=18)
            ax.set_xlabel(train_name, fontsize=18)
            ax.set_ylabel('Sample Count', fontsize=18)
            ax.legend(fontsize=22) #loc='upper left'
            
            for i, (bar, ro) in enumerate(zip(ood_bars, ood_ratios_c)):
                plt.text(bar.get_x()+bar.get_width()/2.0, bar.get_height(), f'{ro:.1f}%', ha='center', va='top', fontsize=20)                     
            plt.tight_layout()
            plt.savefig(join(analysis_path, f'ID_OOD_class{int(threshold * 100)}.png'), dpi=300)
        print(f"analysis saved in '{analysis_path}'")
#--------------------------------------main--------------------------------------------------------
analysis(train_path, save=True)
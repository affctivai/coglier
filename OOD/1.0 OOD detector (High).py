import os
from os.path import join, exists
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.constant import *
from utils.transform import scaling, deshape
from sklearn.model_selection import train_test_split
from utils.dataset import load_list_subjects, PreprocessedDataset
from utils.model import get_model, get_model_with_dropout
from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.tools import plot_scheduler, epoch_time, plot_train_result
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
parser.add_argument("--epoch", dest="epoch", type=int, action="store", default=1) # 1, 50, 100
parser.add_argument("--dropout", dest="dropout", type=float, action="store", default=0, help='0, 0.2, 0.3, 0.5')

parser.add_argument("--column", dest="column", action="store", default="test_acc", help='test_acc, test_loss, roc_auc_score') # 기준 칼럼
parser.add_argument("--cut", type= int, dest="cut", action="store", default="4") # low group count
parser.add_argument("--test", dest="test", action="store_true") # Whether to train data
args = parser.parse_args()

DATASET_NAME = args.dataset
LABEL = args.label
MODEL_NAME = args.model
FEATURE = args.feature
BATCH = args.batch
EPOCH = args.epoch
DROPOUT = args.dropout

PROJECT = 'High'
COLUMN = args.column
CUT = args.cut
TEST = args.test

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

# After subdepend.py-----------------------------------------
def get_ID():
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

HIGS, LOWS = get_ID()

print(train_name, 'HIGS', len(HIGS),'명', HIGS)
print(train_name, 'LOWS', len(LOWS),'명', LOWS)

# Load data
datas, targets = load_list_subjects(DATA, 'train', HIGS, LABEL)
datas = scaling(datas, scaler_name=SCALE)
datas = deshape(datas, shape_name=SHAPE, chls=CHLS, location=LOCATION)

# Split into train, valid, test
X_train, X, Y_train, Y = train_test_split(datas, targets, test_size=0.1, stratify=targets, random_state=random_seed)
X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=random_seed)

trainset = PreprocessedDataset(X_train, Y_train)
validset = PreprocessedDataset(X_valid, Y_valid)
testset = PreprocessedDataset(X_test, Y_test)
print(f'High-trainset: {trainset.x.shape} \t High-validset: {validset.x.shape}')

labels_name = np.unique(validset.y) + 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-------------------------------------train--------------------------------------------------------
def run_train():
    print(f'{DATASET_NAME} {MODEL_NAME} {FEATURE} (shape:{SHAPE},scale:{SCALE}) LABEL:{train_name}')

    trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)

    model, max_lr = get_model_with_dropout(MODEL_NAME, validset.x.shape, len(labels_name), device, DROPOUT)
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    STEP = len(trainloader)
    STEPS = EPOCH * STEP

    optimizer = optim.Adam(model.parameters(), lr=0, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=STEPS, T_mult=1, eta_max=max_lr, T_up=STEP*3, gamma=0.5)
    
    def train(model, loader, optimizer, criterion, scheduler, device, scaler):
        epoch_loss = 0; epoch_acc = 0
        model.train()
        for (x, y, subID) in loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                y_pred = model(x)
                loss = criterion(y_pred, y)
                acc = y.eq(y_pred.argmax(1)).sum() / y.shape[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        return epoch_loss / len(loader), epoch_acc / len(loader)

    def evaluate(model, loader, criterion, device):
        epoch_loss = 0; epoch_acc = 0
        model.eval()
        with torch.no_grad():
            for (x, y, subID) in loader:
                x = x.to(device); y = y.to(device)
                
                y_pred = model(x)
                loss = criterion(y_pred, y)

                acc = y.eq(y_pred.argmax(1)).sum() / y.shape[0]
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss/len(loader), epoch_acc/len(loader)

    train_path.mkdir(parents=True, exist_ok=True)
    with open(join(train_path, 'train.txt'), 'w') as file:
        file.write(f'{train_name} {labels_name} train:{tuple(trainset.x.shape)} valid:{tuple(validset.x.shape)}\n'
                   f'Epoch_{EPOCH}  Train_Loss|Acc\tValid_Loss|Acc\n')
        lrs = []
        train_losses, train_accs, valid_losses, valid_accs = [], [], [], []
        best_valid_loss = float('inf')
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(EPOCH):
            start_time = time.monotonic()
            train_loss, train_acc = train(model, trainloader, optimizer, criterion, scheduler, device, scaler)
            valid_loss, valid_acc = evaluate(model, validloader, criterion, device)

            train_losses.append(train_loss); valid_losses.append(valid_loss)
            train_accs.append(train_acc); valid_accs.append(valid_acc)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), join(train_path,'best.pt'))
            
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            log = f'{epoch+1:03} {epoch_secs:2d}s \t {train_loss:1.3f}\t{train_acc*100:6.2f}%\t{valid_loss:1.3f}\t{valid_acc*100:6.2f}%'
            file.write(log + '\n')
            print(log)
    plot_scheduler(lrs, save=True, path=train_path)
    plot_train_result(train_losses, valid_losses, train_accs, valid_accs, EPOCH, size=(9, 5), path=train_path)
    print(f"model weights saved in '{join(train_path,'best.pt')}'")

#--------------------------------------test--------------------------------------------------------
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

def detect(train_path):
    if not exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}, Set the train weight path properly.")
    
    test_path = Path(join(train_path, 'test'))
    test_path = get_folder(test_path)

    # Load Low subjects
    datas_l, targets_l = load_list_subjects(DATA, 'train', LOWS, LABEL)
    datas_l = scaling(datas_l, scaler_name=SCALE)
    datas_l = deshape(datas_l, shape_name=SHAPE, chls=CHLS, location=LOCATION)
    lowsset = PreprocessedDataset(datas_l, targets_l)
    print(f'testset for measuring OOD performance| Highs: {testset.x.shape}, Lows: {lowsset.x.shape}')

    testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)
    lowsloader = DataLoader(lowsset, batch_size=BATCH, shuffle=False)

    model, _ = get_model_with_dropout(MODEL_NAME, testset.x.shape, len(labels_name), device, DROPOUT)
    model.load_state_dict(torch.load(join(train_path, 'best.pt')))

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)
    
    test_path.mkdir(parents=True, exist_ok=True)
    with open(join(test_path, 'output.txt'), 'w') as file:
        file.write(f'{train_name} {labels_name} high:{tuple(testset.x.shape)} low:{tuple(lowsset.x.shape)}\tcol:{COLUMN}\n'
                   f'LOWS {len(LOWS)}명 {LOWS}\nHIGS {len(HIGS)}명 {HIGS}\n')

        losss, accs, labels, preds, msps_higs, _ = evaluate_test(model, testloader, criterion, device)
        _,        _,      _,     _, msps_lows, _ = evaluate_test(model, lowsloader, criterion, device)
        
        high_loss, high_acc = torch.mean(losss.float()), torch.mean(accs.float())
 
        log = (f'high_loss: {high_loss:.3f}\thigh_acc: {high_acc*100:6.2f}%\troc_auc_score: {get_roc_auc_score(labels, preds)}\n')

        log += '----------OOD detection performance----------\n'
        y_true = torch.cat([torch.ones(len(testset)), torch.zeros(len(lowsset))])
        y_pred = torch.cat([msps_higs, msps_lows])

        log += print_auroc(y_true, y_pred, percent=0.95, path=test_path)

        file.write(log)
        print(log)

        # plot confidence histogram
        plt.figure(figsize=(8,8))
        plt.hist(msps_lows.cpu() , bins=100, alpha=0.4, label='Lows')
        plt.hist(msps_higs.cpu(), bins=100, alpha=0.4, label='Highs')
        plt.xlabel('Confidence', fontsize=15)
        plt.ylabel('num of samples', fontsize=15)
        plt.legend(fontsize=25)
        plt.tight_layout()
        plt.savefig(join(test_path, 'confidence_hist.png'), dpi=200)
    print(f'saved in {test_path}')

def analysis(train_path):
    if not exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}, Set the train weight path properly.")
    
    analysis_path = Path(join(train_path, 'analysis'))
    analysis_path = get_folder(analysis_path)

    SUBLIST = [str(i).zfill(2) for i in range(1, SUB_NUM + 1)]  # '01', '02', ...
    datas, targets = load_list_subjects(DATA, 'train', SUBLIST, LABEL)
    datas = scaling(datas, scaler_name=SCALE)
    datas = deshape(datas, shape_name=SHAPE, chls=CHLS, location=LOCATION)
    dataset = PreprocessedDataset(datas, targets)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
    model, _ = get_model_with_dropout(MODEL_NAME, dataset.x.shape, len(labels_name), device, DROPOUT)
    model.load_state_dict(torch.load(join(train_path, 'best.pt')))
    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)

    _, _, _, _, msps, _ = evaluate_test(model, loader, criterion, device)
    
    analysis_path.mkdir(parents=True, exist_ok=True)
    for threshold in THRESHOLDS:
        ind_idxs = msps >= threshold
        ood_idxs = msps < threshold
        
        n_ind = ind_idxs.sum().item()
        n_ood = len(ind_idxs) - n_ind
        print(f'T:{threshold}\tID/OOD count|ratio : {n_ind},{n_ood}|{n_ind/len(ind_idxs):.2f},{n_ood/len(ind_idxs):.2f}\n')

        datas_ind, targets_ind = datas[ind_idxs], targets[ind_idxs]
        datas_ood, targets_ood = datas[ood_idxs], targets[ood_idxs]

        ## plot ID & OOD per subject
        subids, ind_subs = np.unique(targets_ind[:, 1], return_counts=True)
        _, ood_subs = np.unique(targets_ood[:, 1], return_counts=True)

        plt.figure(figsize=(10, 15))
        plt.bar(subids, ood_subs, color='#4a7fb0', label='OOD')
        plt.bar(subids, ind_subs, color='#adc6e5', bottom=ood_subs, label='ID')  # stacked bar chart

        plt.title(f'ID/OODs per subject (T={threshold})', fontsize=18)
        plt.ylabel('num of samples', fontsize=15)
        plt.xlabel('Subject ID', fontsize=15)
        plt.xticks(subids, fontsize=10)
        plt.legend(fontsize=25, loc='upper left')
        plt.tight_layout()
        plt.savefig(join(analysis_path, f'ID_OOD_subid{int(threshold * 100)}.png'), dpi=200)

        ## plot ID & OOD per class
        _, ind_class = np.unique(targets_ind[:, 0], return_counts=True)
        _, ood_class = np.unique(targets_ood[:, 0], return_counts=True)

        plt.figure(figsize=(8, 15))
        plt.bar(labels_name, ood_class, color='#4a7fb0', label='OOD')
        plt.bar(labels_name, ind_class, color='#adc6e5', bottom=ood_class, label='ID')

        plt.title(f'ID/OODs per class (T={threshold})', fontsize=18)
        plt.ylabel('num of samples', fontsize=15)
        plt.xlabel(train_name, fontsize=15)
        plt.xticks(labels_name, fontsize=15)
        plt.legend(fontsize=25, loc='upper left')
        plt.tight_layout()
        plt.savefig(join(analysis_path, f'ID_OOD_class{int(threshold * 100)}.png'), dpi=200)

#--------------------------------------main--------------------------------------------------------
train_path = get_folder(train_path)
if not TEST: run_train()
detect(train_path)
THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
analysis(train_path)
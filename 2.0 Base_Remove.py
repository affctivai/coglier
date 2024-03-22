import os
from os.path import join, exists
import time
from pathlib import Path
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.constant import *
from utils.transform import scaling, deshape
from sklearn.model_selection import train_test_split
from utils.dataset import load_list_subjects, PreprocessedDataset
from utils.model import get_model
from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.tools import epoch_time
from utils.tools import get_roc_auc_score
from utils.tools import seed_everything, get_folder

random_seed = 42
seed_everything(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO", help='GAMEEMO, SEED, SEED_IV')
parser.add_argument("--label", dest="label", action="store", default="v", help='v, a :GAMEEMO')
parser.add_argument("--model", dest="model", action="store", default="CCNN", help='CCNN, TSC, DGCNN')
parser.add_argument("--feature", dest="feature", action="store", default="DE", help='DE, PSD, raw')
parser.add_argument("--batch", dest="batch", type=int, action="store", default=64)
parser.add_argument("--epoch", dest="epoch", type=int, action="store", default=1) 
parser.add_argument("--dropout", dest="dropout", type=float, action="store", default=0, help='0, 0.2, 0.3, 0.5')
parser.add_argument("--sr", dest="sr", type=int, action="store", default=128, help='128, 200') # Sampling Rate

parser.add_argument("--test", dest="test", action="store_true", help='Whether to train data')
parser.add_argument("--threshold", dest="threshold", type=float, action="store", default=0, help='0.98, 0.95, 0.90, 0.85')

parser.add_argument("--detector", dest="detector", action="store", default="Low_4", help='Low_CUT')
args = parser.parse_args()

DATASET_NAME = args.dataset
LABEL = args.label
MODEL_NAME = args.model
FEATURE = args.feature
BATCH = args.batch
EPOCH = args.epoch
DROPOUT = args.dropout
TEST = args.test
THRESHOLD = args.threshold
SR = args.sr

DETECTOR = args.detector
PROJECT = f'{DETECTOR}_RR{int(THRESHOLD*100)}'

if MODEL_NAME == 'CCNN': SHAPE = 'grid'
elif MODEL_NAME == 'TSC': SHAPE = 'expand'; FEATURE = 'raw'
elif MODEL_NAME == 'DGCNN': SHAPE = None
if FEATURE == 'DE': SCALE = None
elif FEATURE == 'PSD': SCALE = 'log'
elif FEATURE == 'raw': SCALE = 'standard'
if LABEL == 'a':    train_name = 'arousal'
elif LABEL == 'v':  train_name = 'valence'
else:               train_name = 'emotion'
if MODEL_NAME == 'TSC': MODEL_FEATURE = MODEL_NAME
else:  MODEL_FEATURE = '_'.join([MODEL_NAME, FEATURE])

DATAS, SUB_NUM, CHLS, LOCATION = load_dataset_info(DATASET_NAME)
SUBLIST = [str(i).zfill(2) for i in range(1, SUB_NUM+1)]
DATA = join(DATAS, FEATURE)

train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, MODEL_FEATURE, PROJECT, train_name))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def remove_ood(datas, targets, threshold, detoctor_name):
    print('Detector name:', detoctor_name)
    dataset = PreprocessedDataset(datas, targets)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)
    ood_detector_path = Path(join(os.getcwd(), 'results', DATASET_NAME, MODEL_FEATURE, detoctor_name, train_name))
    ood_detector, _ = get_model(MODEL_NAME, dataset.x.shape, len(np.unique(dataset.y)+1), device, sampling_rate=SR)
    ood_detector = ood_detector.to(device)
    ood_detector.load_state_dict(torch.load(join(ood_detector_path, 'best.pt')))
    msps = []
    ood_detector.eval()
    with torch.no_grad():
        for (x, y, subID) in loader:
            x = x.to(device)
            y_pred = ood_detector(x)
            msp = nn.functional.softmax(y_pred, dim=-1)
            msp, maxidx = msp.max(1)
            msps.append(msp.cpu())
    msps = torch.cat(msps, dim=0)
    ind_idxs = msps >= threshold
    n_ind = ind_idxs.sum().item()
    n_ood = len(ind_idxs) - n_ind

    remove_info = f'T:{threshold}\tID/OOD count|ratio : {n_ind},{n_ood}|{n_ind/len(ind_idxs):.2f},{n_ood/len(ind_idxs):.2f}\n'
    print(remove_info)
    datas_ind, targets_ind = datas[ind_idxs], targets[ind_idxs]
    return datas_ind, targets_ind, remove_info

#--------------------------------------train-------------------------------------------------------
def run_train():
    print(f'{DATASET_NAME} {MODEL_NAME} {FEATURE} (shape:{SHAPE},scale:{SCALE}) LABEL:{train_name}')

    datas, targets = load_list_subjects(DATA, 'train', SUBLIST, LABEL)
    datas = scaling(datas, scaler_name=SCALE)
    datas = deshape(datas, shape_name=SHAPE, chls=CHLS, location=LOCATION)
    
    if THRESHOLD > 0: datas, targets, remove_info = remove_ood(datas, targets, THRESHOLD, DETECTOR)

    X_train, X_valid, Y_train, Y_valid = train_test_split(datas, targets, test_size=0.1, stratify=targets, random_state=random_seed)
    
    trainset = PreprocessedDataset(X_train, Y_train)
    validset = PreprocessedDataset(X_valid, Y_valid)
    print(f'trainset: {trainset.x.shape} \t validset: {validset.x.shape}')

    trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)

    labels_name = np.unique(validset.y) + 1

    model, max_lr = get_model(MODEL_NAME, validset.x.shape, len(labels_name), device, DROPOUT, sampling_rate=SR)

    STEP = len(trainloader)
    STEPS = EPOCH * STEP

    optimizer = optim.Adam(model.parameters(), lr=0, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=STEPS, T_mult=1, eta_max=max_lr, T_up=STEP*3, gamma=0.5)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    def top_k_accuracy(y_pred, y, k=1):
        _, indices = y_pred.topk(k, dim=-1)
        correct = indices.eq(y.view(-1, 1))
        return correct.any(dim=-1).sum().item() / y.size(0)
    
    def train(model, loader, optimizer, criterion, scheduler, device, scaler):
        epoch_loss, epoch_acc_1 = 0, 0
        model.train()
        for (x, y, subID) in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                y_pred = model(x)
                loss = criterion(y_pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_acc_1 += top_k_accuracy(y_pred, y, k=1)

            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        epoch_loss /= len(loader); epoch_acc_1 /= len(loader)
        return epoch_loss, epoch_acc_1

    def evaluate(model, loader, criterion, device):
        epoch_loss, epoch_acc_1 = 0, 0
        model.eval()
        with torch.no_grad():
            for (x, y, subID) in loader:
                x, y = x.to(device), y.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc_1 += top_k_accuracy(y_pred, y, k=1)
        epoch_loss /= len(loader); epoch_acc_1 /= len(loader)
        return epoch_loss, epoch_acc_1

    train_path.mkdir(parents=True, exist_ok=True)
    with open(join(train_path, 'train.txt'), 'w') as file:
        file.write(f'{train_name} {labels_name} train:{tuple(trainset.x.shape)} valid:{tuple(validset.x.shape)}\n'
                   f'Epoch_{EPOCH}\tTrain_Loss|Acc\tValid_Loss|Acc\n')
        if THRESHOLD > 0: file.write(remove_info)

        lrs = []
        train_losses, train_accs, valid_losses, valid_accs = [], [], [], []
        best_valid_loss = float('inf')
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(EPOCH):
            start_time = time.monotonic()
            train_loss, train_acc_1 = train(model, trainloader, optimizer, criterion, scheduler, device, scaler)
            valid_loss, valid_acc_1 = evaluate(model, validloader, criterion, device)

            train_losses.append(train_loss); valid_losses.append(valid_loss)
            train_accs.append(train_acc_1); valid_accs.append(valid_acc_1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), join(train_path,'best.pt'))

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            log = (f'{epoch+1:03} {epoch_secs:2d}s\t{train_loss:1.3f}\t{train_acc_1*100:6.2f}%'
                   f'\t\t{valid_loss:1.3f}\t{valid_acc_1*100:6.2f}%')
            file.write(log + '\n')
            print(log)
    print(f"model weights saved in '{join(train_path,'best.pt')}'")

#--------------------------------------test--------------------------------------------------------
def run_test(train_path):
    if not exists(train_path): raise FileNotFoundError(f"File not found: {train_path}, Set the train weight path properly.")
    
    test_path = Path(join(train_path, 'test'))
    test_path = get_folder(test_path)
    
    datas, targets = load_list_subjects(DATA, 'test', SUBLIST, LABEL)
    datas = scaling(datas, scaler_name=SCALE)
    datas = deshape(datas, shape_name=SHAPE, chls=CHLS, location=LOCATION)

    if THRESHOLD > 0: datas, targets, remove_info = remove_ood(datas, targets, THRESHOLD, DETECTOR)
    
    testset = PreprocessedDataset(datas, targets)
    print(f'testset: {testset.x.shape}')
    testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)
    labels_name = np.unique(testset.y) + 1
    model, _ = get_model(MODEL_NAME, testset.x.shape, len(labels_name), device, DROPOUT, sampling_rate=SR)
    model.load_state_dict(torch.load(join(train_path, 'best.pt')))

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)

    def top_k_accuracy(y_pred, y, k=1):
        _, indices = y_pred.topk(k, dim=-1)
        correct = indices.eq(y.view(-1, 1))
        return correct.any(dim=-1)
    
    def evaluate_test(model, loader, criterion, device):
        losss, accs_1 = [], []
        labels, preds = [], []
        model.eval() 
        with torch.no_grad():
            for (x, y, subID) in loader:
                x, y = x.to(device), y.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y)

                msp = nn.functional.softmax(y_pred, dim=-1)
                msp, maxidx = msp.max(1)

                accs_1.append(top_k_accuracy(y_pred, y, k=1).cpu())
                losss.append(loss.cpu())
                labels.append(y.cpu())
                preds.append(maxidx.cpu())
        accs_1 = torch.cat(accs_1, dim=0)
        losss = torch.cat(losss, dim=0)    
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        return losss, accs_1, labels, preds

    test_path.mkdir(parents=True, exist_ok=True)
    with open(join(test_path, 'output.txt'), 'w') as file:
        file.write(f'{train_name} {labels_name} test:{tuple(testset.x.shape)}\n')
        if THRESHOLD > 0: file.write(remove_info)
        
        losss, accs_1, labels, preds  = evaluate_test(model, testloader, criterion, device)
        
        test_loss = torch.mean(losss.float()).item()
        test_acc_1 = torch.mean(accs_1.float()).item()

        log = f'test_loss: {test_loss:.3f}\ttest_acc: {test_acc_1*100:.2f}%:\t'
        log += f'roc_auc_score: {get_roc_auc_score(labels, preds)}\n'
        
        print(log)
    print(f'saved in {test_path}')

train_path = get_folder(train_path)
if not TEST: run_train()
run_test(train_path)
import os
from os.path import join, exists
import time
from pathlib import Path
import numpy as np
import argparse

import torch
# from torchsummary import summary
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.constant import *
from utils.transform import scaling, deshape
from sklearn.model_selection import train_test_split
from utils.dataset import load_list_subjects_rp, PreprocessedDataset
from utils.model import get_model, get_model_with_dropout
from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.tools import MyScheduler, plot_scheduler, epoch_time, plot_train_result
from utils.tools import plot_confusion_matrix, get_roc_auc_score
from utils.tools import seed_everything, get_folder
from sklearn.metrics import classification_report

random_seed = 42
seed_everything(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO", help='GAMEEMO, SEED, SEED_IV, DEAP')
parser.add_argument("--label", dest="label", action="store", default="v", help='v, a :GAMEEMO/DEAP')
parser.add_argument("--model", dest="model", action="store", default="CCNN", help='CCNN, TSC, EEGNet, DGCNN')
parser.add_argument("--feature", dest="feature", action="store", default="DE", help='DE, PSD, raw')
parser.add_argument("--batch", dest="batch", type=int, action="store", default=64)
parser.add_argument("--epoch", dest="epoch", type=int, action="store", default=1) 
parser.add_argument("--dropout", dest="dropout", type=float, action="store", default=0, help='0, 0.2, 0.3, 0.5')

parser.add_argument("--test", dest="test", action="store_true", help='Whether to train data')
parser.add_argument("--topk", dest="topk", type=int, action="store", default=2)
args = parser.parse_args()

DATASET_NAME = args.dataset
LABEL = args.label
MODEL_NAME = args.model
FEATURE = args.feature
BATCH = args.batch
EPOCH = args.epoch
DROPOUT = args.dropout
TEST = args.test
TOPK = args.topk

PROJECT = f'Base_RP'

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
else:  MODEL_FEATURE = '_'.join([MODEL_NAME, FEATURE])

DATAS, SUB_NUM, CHLS, LOCATION = load_dataset_info(DATASET_NAME)
SUBLIST = [str(i).zfill(2) for i in range(1, SUB_NUM+1)] # '01', '02', '03', ...
DATA = join(DATAS, FEATURE)

train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, MODEL_FEATURE, PROJECT, train_name))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#--------------------------------------train-------------------------------------------------------
def run_train():
    print(f'{DATASET_NAME} {MODEL_NAME} {FEATURE} (shape:{SHAPE},scale:{SCALE}) LABEL:{train_name}')

    # Load train data
    datas, targets = load_list_subjects_rp(DATA, 'train', SUBLIST, LABEL)

    # online transform
    datas = scaling(datas, scaler_name=SCALE)
    datas = deshape(datas, shape_name=SHAPE, chls=CHLS, location=LOCATION)

    # Split into train, valid
    X_train, X_valid, Y_train, Y_valid = train_test_split(datas, targets, test_size=0.1, stratify=targets, random_state=random_seed)
    
    trainset = PreprocessedDataset(X_train, Y_train)
    validset = PreprocessedDataset(X_valid, Y_valid)
    print(f'trainset: {trainset.x.shape} \t validset: {validset.x.shape}')

    trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)

    labels_name = np.unique(validset.y) + 1

    # Model
    model, max_lr = get_model_with_dropout(MODEL_NAME, validset.x.shape, len(labels_name), device, DROPOUT)

    STEP = len(trainloader)
    STEPS = EPOCH * STEP

    optimizer = optim.Adam(model.parameters(), lr=0, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=STEPS, T_mult=1, eta_max=max_lr, T_up=STEP*3, gamma=0.5)

    ## plot_scheduler
    # scheduler = MyScheduler(STEP, STEPS, flag='CWR', baselr=0.0001, T_0_num= 1, T_up_num= 3)
    # plot_scheduler(scheduler.make_lrs(), save=True, path=train_path)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    def top_k_accuracy(y_pred, y, k=1):
        _, indices = y_pred.topk(k, dim=-1)
        correct = indices.eq(y.view(-1, 1))
        return correct.any(dim=-1).sum().item() / y.size(0)
    
    def train(model, loader, optimizer, criterion, scheduler, device, scaler, topk):
        epoch_loss, epoch_acc_1, epoch_acc_k = 0, 0, 0
        model.train()
        for (x, y, subID) in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): # AMP
                y_pred = model(x)
                loss = criterion(y_pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_acc_1 += top_k_accuracy(y_pred, y, k=1)
            epoch_acc_k += top_k_accuracy(y_pred, y, k=topk)

            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        epoch_loss /= len(loader); epoch_acc_1 /= len(loader); epoch_acc_k /= len(loader)
        return epoch_loss, epoch_acc_1, epoch_acc_k

    def evaluate(model, loader, criterion, device, topk):
        epoch_loss, epoch_acc_1, epoch_acc_k = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for (x, y, subID) in loader:
                x, y = x.to(device), y.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc_1 += top_k_accuracy(y_pred, y, k=1)
                epoch_acc_k += top_k_accuracy(y_pred, y, k=topk)
        epoch_loss /= len(loader); epoch_acc_1 /= len(loader); epoch_acc_k /= len(loader)
        return epoch_loss, epoch_acc_1, epoch_acc_k

    train_path.mkdir(parents=True, exist_ok=True)
    with open(join(train_path, 'train.txt'), 'w') as file:
        file.write(f'{train_name} {labels_name} train:{tuple(trainset.x.shape)} valid:{tuple(validset.x.shape)}\n'
                   f'Epoch_{EPOCH}\tTrain_Loss|Acc1_Acc{TOPK}\tValid_Loss|Acc1_Acc{TOPK}\n')

        lrs = []
        train_losses, train_accs, valid_losses, valid_accs = [], [], [], []
        best_valid_loss = float('inf')
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(EPOCH):
            start_time = time.monotonic()
            train_loss, train_acc_1, train_acc_k = train(model, trainloader, optimizer, criterion, scheduler, device, scaler, TOPK)
            valid_loss, valid_acc_1, valid_acc_k = evaluate(model, validloader, criterion, device, TOPK)

            train_losses.append(train_loss); valid_losses.append(valid_loss)
            train_accs.append(train_acc_1); valid_accs.append(valid_acc_1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), join(train_path,'best.pt'))

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            log = (f'{epoch+1:03} {epoch_secs:2d}s\t{train_loss:1.3f}\t{train_acc_1*100:6.2f}%\t{train_acc_k*100:6.2f}%'
                   f'\t\t{valid_loss:1.3f}\t{valid_acc_1*100:6.2f}%\t{valid_acc_k*100:6.2f}%')
            file.write(log + '\n')
            print(log)
    plot_scheduler(lrs, save=True, path=train_path)
    plot_train_result(train_losses, valid_losses, train_accs, valid_accs, EPOCH, size=(9, 5), path=train_path)
    print(f"model weights saved in '{join(train_path,'best.pt')}'")

#--------------------------------------test--------------------------------------------------------
def run_test(train_path):
    if not exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}, Set the train weight path properly.")
    
    test_path = Path(join(train_path, 'test'))
    test_path = get_folder(test_path)
    
    # Load test data
    datas, targets = load_list_subjects_rp(DATA, 'test', SUBLIST, LABEL)
    # online transform
    datas = scaling(datas, scaler_name=SCALE)
    datas = deshape(datas, shape_name=SHAPE, chls=CHLS, location=LOCATION)

    testset = PreprocessedDataset(datas, targets)
    print(f'testset: {testset.x.shape}')

    testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)

    labels_name = np.unique(testset.y) + 1

    # Model (load parameters)
    model, _ = get_model_with_dropout(MODEL_NAME, testset.x.shape, len(labels_name), device, DROPOUT)
    model.load_state_dict(torch.load(join(train_path, 'best.pt')))

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = criterion.to(device)

    def top_k_accuracy(y_pred, y, k=1):
        _, indices = y_pred.topk(k, dim=-1)
        correct = indices.eq(y.view(-1, 1))
        return correct.any(dim=-1)
    
    def evaluate_test(model, loader, criterion, device, topk):
        losss, accs_1, accs_k = [], [], []
        labels, preds, msps, subIDs = [], [], [], []
        model.eval() 
        with torch.no_grad():
            for (x, y, subID) in loader:
                x, y = x.to(device), y.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y)

                msp = nn.functional.softmax(y_pred, dim=-1)
                msp, maxidx = msp.max(1)

                accs_1.append(top_k_accuracy(y_pred, y, k=1).cpu())
                accs_k.append(top_k_accuracy(y_pred, y, k=topk).cpu())
                losss.append(loss.cpu())
                labels.append(y.cpu())
                preds.append(maxidx.cpu())
                msps.append(msp.cpu())
                subIDs.append(subID.cpu())
        accs_1 = torch.cat(accs_1, dim=0)
        accs_k = torch.cat(accs_k, dim=0)
        losss = torch.cat(losss, dim=0)    
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        msps = torch.cat(msps, dim=0)
        subIDs = torch.cat(subIDs, dim=0)
        return losss, accs_1, accs_k, labels, preds, msps, subIDs

    # ----------------------------------------run-------------------------------------------------------
    test_path.mkdir(parents=True, exist_ok=True)
    with open(join(test_path, 'output.txt'), 'w') as file:
        file.write(f'{train_name} {labels_name} test:{tuple(testset.x.shape)}\n')
        
        losss, accs_1, accs_k, labels, preds, msps, subIDs  = evaluate_test(model, testloader, criterion, device, TOPK)
        
        # ----------OOD detection----------
        corrects = accs_1
        
        test_loss = torch.mean(losss.float()).item()
        test_acc_1, test_acc_k = torch.mean(accs_1.float()).item(), torch.mean(accs_k.float()).item()

        log = f'test_loss: {test_loss:.3f}\ttest_acc_1: {test_acc_1*100:.2f}%\ttest_acc_{TOPK}: {test_acc_k*100:.2f}%\t'
        log += f'roc_auc_score: {get_roc_auc_score(labels, preds)}\n'
        log += classification_report(labels, preds)
        
        log += f'\n-----Accuracy by subject-----'        
        n_tests     = torch.bincount(subIDs)[1:]
        n_corrects  = torch.bincount(subIDs, corrects).int()[1:]

        unique_ids = np.unique(testset.subID)
        log += f'\nSubID\tacc\tn_correct\tn_test\n'
        for sub, n_test, n_corr in zip(unique_ids, n_tests, n_corrects):
            acc = (n_corr / n_test).item()
            log += f'{sub:03d}\t{acc:.4f}\t{n_corr.item():6d}\t{n_test.item():6d}\n'
        file.write(log)
        print(log)
    plot_confusion_matrix(labels, preds, labels_name, path=test_path, lbname=train_name, title=f'{train_name} CM')
    print(f'saved in {test_path}')

#---------------------------------------main-------------------------------------------------------
train_path = get_folder(train_path)
if not TEST: run_train()
run_test(train_path)
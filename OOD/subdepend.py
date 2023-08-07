import os
from os.path import join
import time
import random
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
from utils.dataset import load_per_subject, PreprocessedDataset_
from utils.model import CCNN, TSCeption, EEGNet, DGCNN
from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.tools import plot_scheduler, epoch_time, plot_train_result
from utils.tools import plot_confusion_matrix, get_roc_auc_score
import math

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
random_seed = 42
seed_everything(random_seed)

#-----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", default="/mnt/data/research_EG", help='After 0.0 preprocessing.py')
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO", help='GAMEEMO, SEED, SEED_IV, DEAP')
parser.add_argument('--subID', type=str, default = '01')
parser.add_argument("--label", type=str, default='v', help='v, a :GAMEEMO/DEAP')
parser.add_argument("--model", dest="model", action="store", default="CCNN", help='CCNN, TSC, EEGNet, DGCNN')
parser.add_argument("--feature", dest="feature", action="store", default="DE", help='DE, PSD, raw')
parser.add_argument('--batch', type=int, default = 64)
parser.add_argument('--epoch', type=int, default = 3)
parser.add_argument('--project_name', type=str, default = 'Subdepend')  # save result
args = parser.parse_args()

DATASETS = args.datasets
DATASET_NAME = args.dataset
SUB = args.subID
LABEL = args.label
MODEL_NAME = args.model
FEATURE = args.feature
BATCH = args.batch
EPOCH = args.epoch
PROJECT = args.project_name

if DATASET_NAME == 'GAMEEMO':
    DATAS = join(DATASETS, 'GAMEEMO_npz', 'Projects')
    SUB_NUM = GAMEEMO_SUBNUM
    CHLS = GAMEEMO_CHLS
    LOCATION = GAMEEMO_LOCATION
elif DATASET_NAME == 'SEED':
    DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz","Projects")
    # LABEL = '4' # 4, v, a
    # EPOCH = 1
    # BATCH = 128
elif DATASET_NAME == 'SEED_IV':
    DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")
    # LABEL = '4' # 4, v, a
    # EPOCH = 100
    # BATCH = 128
elif DATASET_NAME == 'DEAP':
    DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")
    # LABEL = 'v' # 4, v, a
    # EPOCH = 1
    # BATCH = 64
else:
    print("Unknown Dataset")
    exit(1)

if MODEL_NAME == 'CCNN': SHAPE = 'grid'
elif MODEL_NAME == 'TSC' or MODEL_NAME == 'EEGNet':
    SHAPE = 'expand'
    FEATURE = 'raw'
elif MODEL_NAME == 'DGCNN': SHAPE = None

if FEATURE == 'DE': SCALE = None
elif FEATURE == 'PSD': SCALE = 'log'
elif FEATURE == 'raw': SCALE = 'standard'

if LABEL == 'a':    train_name = 'arousal'
elif LABEL == 'v':  train_name = 'valence'
else:               train_name = 'emotion'

DATA = join(DATAS, FEATURE)
SUBLIST = [str(i).zfill(2) for i in range(1, SUB_NUM+1)] # '01', '02', '03', ...

#-------------------------------------------------train---------------------------------------------------------------
# Load train data
datas, targets = load_per_subject(DATA, 'train', SUB, LABEL)

# online transform
datas = scaling(datas, scaler_name=SCALE)
datas = deshape(datas, shape_name=SHAPE, chls=CHLS, location=LOCATION)

# Split into train, valid
X_train, X_valid, Y_train, Y_valid = train_test_split(datas, targets, test_size=0.1, stratify=targets, random_state=random_seed)

trainset = PreprocessedDataset_(X_train, Y_train)
validset = PreprocessedDataset_(X_valid, Y_valid)
print(f'trainset: {trainset.x.shape} \t validset: {validset.x.shape}')

trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)

labels_name = validset.label.tolist()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
if MODEL_NAME == 'CCNN':
    model = CCNN(num_classes=len(labels_name), dropout=0.5)
    max_lr = 1e-4
elif MODEL_NAME == 'TSC':
    model = TSCeption(num_electrodes=trainset.x.shape[2], num_classes=len(labels_name), sampling_rate=128, dropout=0.5)
    max_lr = 1e-3
elif MODEL_NAME == 'EEGNet':
    model = EEGNet(chunk_size=trainset.x.shape[3], num_electrodes=trainset.x.shape[2], num_classes=len(labels_name), dropout=0.5)
    max_lr = 1e-3
elif MODEL_NAME == 'DGCNN':
    model = DGCNN(in_channels=trainset.x.shape[2], num_electrodes=trainset.x.shape[1], num_classes=len(labels_name))
    max_lr = 1e-3

model = model.to(device)
# print(summary(model, trainset.x.shape[1:]))

STEP = len(trainloader)
STEPS = EPOCH * STEP

optimizer = optim.Adam(model.parameters(), lr=0, weight_decay=1e-4)
scheduler = CosineAnnealingWarmUpRestarts(optimizer,T_0=STEPS,T_mult=1,eta_max=max_lr,T_up=STEP*3,gamma=0.5)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

def train(model, loader, optimizer, criterion, scheduler, device, scaler):
    epoch_loss = 0; epoch_acc = 0
    model.train()
    for (x, y) in loader:
        x = x.to(device);   y = y.to(device)
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
        for (x, y) in loader:
            x = x.to(device);   y = y.to(device)
            with torch.cuda.amp.autocast():
                y_pred = model(x)
                loss = criterion(y_pred, y)
                acc = y.eq(y_pred.argmax(1)).sum() / y.shape[0]
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss/len(loader), epoch_acc/len(loader)


lrs = []
train_losses, train_accs = [],[]
valid_losses, valid_accs = [],[]
best_valid_loss = float('inf')
scaler = torch.cuda.amp.GradScaler()
# ----------------------------------------run-------------------------------------------------------
train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, PROJECT, SUB, train_name))
train_path.mkdir(parents=True, exist_ok=True)
with open(join(train_path, 'train.txt'), 'w') as file:
    file.write(f'{train_name} {labels_name} train:{tuple(trainset.x.shape)} valid:{tuple(validset.x.shape)}\n'
               f'Epoch {EPOCH}  Train  Loss/Acc\tValid  Loss/Acc\n')
    print(f'Epoch {EPOCH}\tTrain  Loss/Acc\tValid  Loss/Acc')
    for epoch in range(EPOCH):
        start_time = time.monotonic()
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, scheduler, device, scaler)
        valid_loss, valid_acc = evaluate(model, validloader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), join(train_path, 'best.pt'))

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        log = f'{epoch+1:02} {epoch_secs:2d}s \t {train_loss:1.3f}\t{train_acc*100:6.2f}%\t{valid_loss:1.3f}\t{valid_acc*100:6.2f}%'
        file.write(log + '\n')
        print(log)
# plot_scheduler(lrs, save=True, path=train_path)
plot_train_result(train_losses, valid_losses, train_accs, valid_accs, EPOCH, size=(9, 5), path=train_path)
print(f"model weights saved in '{join(train_path,'best.pt')}'")

#--------------------------------------test-------------------------------------------------------
model.load_state_dict(torch.load(join(train_path, 'best.pt')))

# Load test data
datas, targets = load_per_subject(DATA, 'test', SUB, LABEL)
# online transform
datas = scaling(datas, scaler_name=SCALE)
datas = deshape(datas, shape_name=SHAPE, chls=CHLS, location=LOCATION)
testset = PreprocessedDataset_(datas, targets)
print(f'testset: {testset.x.shape}')
testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

def evaluate_test(model, loader, criterion, device):
    losss, accs = [], []
    labels, preds = [], []
    model.eval()
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device);   y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = y.eq(y_pred.argmax(1)).sum() / y.shape[0]
            losss.append(loss.item())
            accs.append(acc.item())
            labels.append(y.cpu().int())
            preds.append(y_pred.argmax(1).cpu())
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    return np.mean(losss), np.mean(accs), labels, preds

with open(join(train_path, 'test.txt'), 'w') as file:
    test_loss, test_acc, labels, preds  = evaluate_test(model, testloader, criterion, device)

    log = f"'test_loss':{test_loss:.3f},'test_acc':{test_acc*100:.2f},"
    log += f"'roc_auc_score':{get_roc_auc_score(labels, preds)}"
    # log += '\n'+classification_report(labels, preds)
    file.write(log)
    # file.write(f'{LABEL}:{labels_name}\t BATCH {BATCH}\n{DNAME}  testset:{tuple(testset.x.shape)}\n')
    print(log)

plot_confusion_matrix(labels, preds, labels_name, path=train_path, lbname=train_name, title=f'Subject{SUB} {train_name}')
print(f'saved in {train_path}')
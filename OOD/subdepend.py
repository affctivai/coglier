import os
from os.path import join, exists
import time
import random
from pathlib import Path
import numpy as np

import torch
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.dataset import GameemoDataset_
from utils.model import MyCCNN
from utils.tools import MyScheduler, plot_scheduler, epoch_time, plot_train_result
from torch.optim.lr_scheduler import _LRScheduler
import math
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
random_seed = 42
seed_everything(random_seed)

def get_folder(path):
    if path.exists():
        for n in range(2, 100):
            p = f'{path}{n}'
            if not exists(p):
                break
        path = Path(p)
    return path
#-----------------------------------------------------------------------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--subID', type=str, default = '01')
parser.add_argument('--epoch', type=int, default = 3)
parser.add_argument('--batch', type=int, default = 64)
parser.add_argument('--target', type=str, default = 'v') # 4, v, a
parser.add_argument('--project_name', type=str, default = 'Subdepen_project')  # save result
args = parser.parse_args()

SUB   = args.subID
EPOCH = args.epoch
BATCH = args.batch
LABEL = args.target
projcet_name = args.project_name


# ---- GAMEEMO
# DATASET_NAME = "GAMEEMO"
# DATAS = join("C:\\", "Users", "LAPTOP", "jupydir", "DATAS", 'GAMEEMO_npz', 'Projects')


# ---- DEAP
# DATASET_NAME = "DEAP"
# DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")

# ---- SEED_IV
DATASET_NAME = "SEED_IV"
DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")

# ---- SEED
# DATASET_NAME = "SEED"
# DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")



project_name = 'subdepend_de'
# project_name = 'subdepend_TSC'

DATA = join(DATAS, project_name, SUB)
DNAME = 'seg_DE'
NAME = f'{DNAME}_{LABEL}'

if LABEL == 'a': train_name = 'arousal'
elif LABEL == 'v': train_name = 'valence'
else: train_name = 'emotion'

train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, project_name, SUB, train_name))
# train_path = get_folder(train_path)
train_path.mkdir(parents=True, exist_ok=True)

# Load train, valid
trainset = GameemoDataset_(DATA, NAME, 'train')
validset = GameemoDataset_(DATA, NAME, 'valid')
print(f'trainset: {trainset.x.shape} \t validset: {validset.x.shape}')

labels_name = validset.label.tolist()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = MyCCNN(in_channels=trainset.x.shape[1], num_classes=len(labels_name))
model = model.to(device)
# print(summary(model, trainset.x.shape[1:]))

trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)

STEP = len(trainloader)
STEPS = EPOCH * STEP

#--------------------------------------train-------------------------------------------------------
def train(model, loader, optimizer, criterion, scheduler, device, scaler):
    epoch_loss = 0; epoch_acc = 0;
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

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=0)
scheduler = CosineAnnealingWarmUpRestarts(optimizer,T_0=STEPS,T_mult=1,eta_max=0.0001,T_up=STEP*3,gamma=0.5)

lrs = []
train_losses, train_accs = [],[]
valid_losses, valid_accs = [],[]
best_valid_loss = float('inf')
scaler = torch.cuda.amp.GradScaler()
with open(join(train_path, 'train.txt'), 'w') as file:
    file.write(f'LABEL {LABEL}:{labels_name}\t BATCH {BATCH}\
                \n{DNAME}  train:{tuple(trainset.x.shape)}\tvalid:{tuple(validset.x.shape)}\
                \nEpoch {EPOCH}\tTrain  Loss/Acc\tValid  Loss/Acc\n')

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

plot_scheduler(lrs, save=True, path=train_path)
plot_train_result(train_losses, valid_losses, train_accs, valid_accs, EPOCH, size=(9, 5), path=train_path)
print(f"model weights saved in '{join(train_path,'best.pt')}'")

#--------------------------------------test-------------------------------------------------------
from utils.tools import plot_confusion_matrix, get_roc_auc_score
from sklearn.metrics import classification_report

model.load_state_dict(torch.load(join(train_path, 'best.pt')))

testset = GameemoDataset_(DATA, NAME, 'test')
print(f'testset: {testset.x.shape}')
testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)

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

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

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
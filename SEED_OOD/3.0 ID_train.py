import os
from os.path import join, exists
import time
import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

from utils.dataset import GameemoDataset
from utils.model import MyCCNN
from utils.tools import epoch_time, plot_train_result
from utils.scheduler import CosineAnnealingWarmUpRestarts

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
DATAS = join(os.getcwd(),"SEED", 'npz')
Project_name = 'IND'
DATA = join(DATAS, 'IND_OOD')
LABEL = '4' # 4, v, a
DNAME = 'ind'
threshold = 98 # 85, 90, 95
NAME = f'{DNAME}_{LABEL}_{threshold}'
if LABEL == 'a':    train_name = f'arousal_{threshold}'
elif LABEL == 'v':    train_name = f'valence_{threshold}'
else: train_name = f'emotion_{threshold}'

train_path = Path(join(os.getcwd(), 'results', Project_name, train_name)) # save result in train_path
train_path = get_folder(train_path)
train_path.mkdir(parents=True, exist_ok=True)

# Load train, valid
trainset = GameemoDataset(DATA, NAME, 'train')
validset = GameemoDataset(DATA, NAME, 'valid')
print(f'trainset: {trainset.x.shape} \t validset: {validset.x.shape}')

labels_name = np.unique(validset.y) + 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = MyCCNN(in_channels=trainset.x.shape[1], num_classes=len(labels_name))
print(trainset.x.shape[1])
model = model.to(device)

EPOCH = 50
BATCH = 128

trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)

STEP = len(trainloader)
STEPS = EPOCH * STEP
#--------------------------------------train-------------------------------------------------------
def train(model, loader, optimizer, criterion, scheduler, device, scaler):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for (x, y, subID) in loader:
        x = x.to(device)
        print(x.size())
        y = y.to(device)
        optimizer.zero_grad()
        # AMP
        with torch.cuda.amp.autocast():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = y.eq(y_pred.argmax(1)).sum() / y.shape[0]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, loader, criterion, device):
    epoch_loss = 0; epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (x, y, subID) in loader:
            x = x.to(device)
            y = y.to(device)
             #AMP
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

train_losses, train_accs = [], []
valid_losses, valid_accs = [], []
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
            torch.save(model.state_dict(), join(train_path,'best.pt'))

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        log = f'{epoch+1:02} {epoch_secs:2d}s \t {train_loss:1.3f}\t{train_acc*100:6.2f}%\t{valid_loss:1.3f}\t{valid_acc*100:6.2f}%'
        file.write(log + '\n')
        print(log)

plot_train_result(train_losses, valid_losses, train_accs, valid_accs, EPOCH, size=(9, 5), path=train_path)
print(f"model weights saved in '{join(train_path,'best.pt')}'")
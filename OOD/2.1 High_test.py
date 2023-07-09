import os
from os.path import join, exists
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import GameemoDataset
from utils.model import MyCCNN
from utils.tools import get_roc_auc_score, print_auroc

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


Project_name = 'Highs'
DATA = join(DATAS, 'Highs')
LABEL = '4'  # 4, v, a

DNAME = 'seg_DE'
NAME = f'{DNAME}_{LABEL}'

if LABEL == 'a': train_name = 'arousal'
elif LABEL == 'v': train_name = 'valence'
else: train_name = 'emotion'

train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, Project_name, train_name)) # # where the train results folder is located
test_path = Path(join(train_path), 'test')
test_path = get_folder(test_path)
test_path.mkdir(parents=True, exist_ok=True)

# Load test, Lows
testset = GameemoDataset(DATA, NAME, 'test')
lowsset = GameemoDataset(DATA, NAME, 'lows')
print(f'testset: {testset.x.shape}, lowsset: {lowsset.x.shape}')

labels_name = np.unique(testset.y) + 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model
model = MyCCNN(in_channels=testset.x.shape[1], num_classes=len(labels_name))
model = model.to(device)
model.load_state_dict(torch.load(join(train_path, 'best.pt')))
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

BATCH = 128

# evaluate
def evaluate_test(model, loader, criterion, device):
    losss, accs = [], []
    labels, preds = [], []
    msps = []
    model.eval()
    with torch.no_grad():
        for (x, y, subID) in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            msp = F.softmax(y_pred, dim=-1)
            msp, maxidx = msp.max(1)

            acc = y.eq(maxidx).sum() / y.shape[0]
            accs.append(acc.item())
            loss = criterion(y_pred, y)
            losss.append(loss.item())

            labels.append(y.cpu().int())
            preds.append(maxidx.cpu())
            msps.append(msp.cpu())

    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    msps = torch.cat(msps, dim=0)
    return np.mean(losss), np.mean(accs), labels, preds, msps


# High test & Lows detect
def detect():
    with open(join(test_path, 'output.txt'), 'w') as file:
        file.write(f'{LABEL}:{labels_name}\t BATCH {BATCH}\
                    \n{DNAME}  test:{tuple(testset.x.shape)}\n')

        testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)
        lowsloader = DataLoader(lowsset, batch_size=BATCH, shuffle=False)

        high_loss, high_acc, labels, preds, probs_higs  = evaluate_test(model, testloader, criterion, device)
        _, _, _, _, probs_lows = evaluate_test(model, lowsloader, criterion, device)

        log = f'high_loss: {high_loss:.3f}\thigh_acc: {high_acc*100:6.2f}%\n'
        log += f'roc_auc_score: {get_roc_auc_score(labels, preds)}\n'

        y_true = torch.cat([torch.ones(len(testset)), torch.zeros(len(lowsset))])
        y_pred = torch.cat([probs_higs, probs_lows])

        log += print_auroc(y_true, y_pred, percent=0.95, path=test_path)

        file.write(log)
        print(log)
        # plot confidence histogram
        plt.figure(figsize=(8,8))
        plt.hist(probs_lows.cpu() , bins=100, alpha=0.4, label='Lows')
        plt.hist(probs_higs.cpu(), bins=100, alpha=0.4, label='Highs')
        plt.xlabel('Confidence', fontsize=15)
        plt.ylabel('num of samples', fontsize=15)
        plt.legend(fontsize=25)
        plt.tight_layout()
        plt.savefig(join(test_path, 'confidence_hist.png'), dpi=200)
    print(f'saved in {test_path}')


# Split ID/OOD
def split_(msps, label, datas, targets, threshold, save=True, analysis=True):
    # id if higher than threshold, ood if lower
    ind_idxs = msps >= threshold
    ood_idxs = msps <  threshold

    _, total_OOD_ID = np.unique(ind_idxs, return_counts=True) # False:OOD, True:ID
    print(f'T:{threshold}\tOOD\tID\ncount{total_OOD_ID} \nratio{np.round(total_OOD_ID/len(msps),2)}')

    datas_ind, targets_ind = datas[ind_idxs], targets[ind_idxs]
    datas_ood, targets_ood = datas[ood_idxs], targets[ood_idxs]
    
    # save data------------------------------------------------------
    if save:
        from sklearn.model_selection import train_test_split

        save_foler = join(DATAS, 'IND_OOD')
        os.makedirs(save_foler, exist_ok=True)
        # save OOD data
        np.savez(join(save_foler, f'ood_{label}_{int(threshold*100)}'), x = datas_ood, y = targets_ood)

        # make ID Dataset  ## train 80 : valid 10 : test 10
        # X_train, X, Y_train, Y = train_test_split(datas_ind, targets_ind, test_size=0.2, stratify=targets_ind, random_state=random_seed)
        # X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=random_seed)
        X_train, X, Y_train, Y = train_test_split(datas_ind, targets_ind, test_size=0.2, random_state=random_seed)
        X_valid, X_test, Y_valid, Y_test = train_test_split(X, Y, test_size=0.5, random_state=random_seed)
        print(f'ID train: {len(Y_train)} \t ID valid: {len(Y_valid)} \t ID test: {len(Y_test)}\n')
        # save ID dataset
        np.savez(join(save_foler, f'ind_{label}_{int(threshold * 100)}_train'), X = X_train, Y = Y_train)
        np.savez(join(save_foler, f'ind_{label}_{int(threshold * 100)}_valid'), X = X_valid, Y = Y_valid)
        np.savez(join(save_foler, f'ind_{label}_{int(threshold * 100)}_test') , X = X_test , Y = Y_test)
        print(f'saved in {save_foler}')

    # Analysis-------------------------------------------------------
    if analysis:
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
        plt.savefig(join(test_path, f'ID_OOD_subid{threshold*100}.png'), dpi=200)

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
        plt.savefig(join(test_path, f'ID_OOD_class{threshold*100}.png'), dpi=200)

# Load all data and divide by ID and OOD
def split(save, analysis):
    from utils.tools import getFromnpz
    from utils.transform import make_grid
    from torch.utils.data import Dataset

    SUBLIST = [str(i) for i in range(1, 16)] # '01', '02', '03', ..., '28'
    datas, targets = getFromnpz(join(DATAS, DNAME), SUBLIST, out=False, cla='4')
    datas = make_grid(datas)  # (33264, 4, 9, 9)

    class tmpDataset(Dataset):
        def __init__(self, X, Y):
            self.x = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(Y[:, 0], dtype=torch.int64)
            self.subID = Y[:, 1]

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.subID[idx]

        def __len__(self):
            return self.y.shape[0]

    allset = tmpDataset(datas, targets)
    allloader = DataLoader(allset, batch_size=BATCH, shuffle=False)
    _, _, _, _, MSPs = evaluate_test(model, allloader, criterion, device)

    split_(MSPs, LABEL, datas, targets, threshold = 0.85, save=save, analysis=analysis)
    split_(MSPs, LABEL, datas, targets, threshold = 0.90, save=save, analysis=analysis)
    split_(MSPs, LABEL, datas, targets, threshold = 0.95, save=save, analysis=analysis)
    split_(MSPs, LABEL, datas, targets, threshold=0.98, save=save, analysis=analysis)

# ------------------------------------main---------------------------------------------
detect()
split(save=True, analysis=True)






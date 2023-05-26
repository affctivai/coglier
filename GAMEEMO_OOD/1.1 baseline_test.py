import os
from os.path import join, exists
import random
from pathlib import Path
import numpy as np

import torch
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import GameemoDataset
from utils.model import MyCCNN
from utils.tools import plot_confusion_matrix, get_roc_auc_score
from sklearn.metrics import classification_report

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
DATAS = join("C:\\", "Users", "LAPTOP", "jupydir", "DATAS", 'GAMEEMO_npz')
DATA = join(DATAS, 'baseline')
LABEL = 'v'  # 4, v, a
DNAME = 'seg_DE'
NAME = f'{DNAME}_{LABEL}'
if LABEL == 'a': train_name = 'arousal'
elif LABEL == 'v': train_name = 'valence'
train_path = join(os.getcwd(), 'results', 'baseline', train_name) # where the train results folder is located
test_path = Path(join(train_path), 'test')
test_path = get_folder(test_path)
test_path.mkdir(parents=True, exist_ok=True)

# Load test
testset = GameemoDataset(DATA, NAME, 'test')
print(f'testset: {testset.x.shape}')

labels_name = np.unique(testset.y) + 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = MyCCNN(in_channels=testset.x.shape[1], num_classes=len(labels_name))
model = model.to(device)
model.load_state_dict(torch.load(join(train_path, 'best.pt')))
# print(summary(model, trainset.x.shape[1:]))

BATCH = 64
testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)
#--------------------------------------test-------------------------------------------------------
def evaluate_test(model, loader, criterion, device):
    losss, accs = [], []
    labels, preds = [], []
    probs, subIDs = [], []
    model.eval()
    with torch.no_grad():
        for (x, y, subID) in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim=-1)
            y_prob, maxidx = y_prob.max(1)

            acc = y.eq(maxidx).sum() / y.shape[0]
            accs.append(acc.item())
            loss = criterion(y_pred, y)
            losss.append(loss.item())

            labels.append(y.cpu().int())
            preds.append(maxidx.cpu())
            probs.append(y_prob.cpu())
            subIDs.append(subID.cpu())

    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    probs = torch.cat(probs, dim=0)
    subIDs = torch.cat(subIDs, dim=0)
    return np.mean(losss), np.mean(accs), labels, preds, probs, subIDs

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

with open(join(test_path, 'output.txt'), 'w') as file:
    file.write(f'{LABEL}:{labels_name}\t BATCH {BATCH}\
                \n{DNAME}  test:{tuple(testset.x.shape)}\n')
    test_loss, test_acc, labels, preds, probs, subIDs  = evaluate_test(model, testloader, criterion, device)
    log = f'test_loss: {test_loss:.3f}\ttest_acc: {test_acc*100:6.2f}%\n'
    log += f'roc_auc_score: {get_roc_auc_score(labels, preds)}\n'
    log += classification_report(labels, preds)

    corrects = torch.eq(labels, preds)
    incorrects_subs = [subID for subID, correct in zip(subIDs, corrects) if not correct]
    unique_sub, test_count = np.unique(testset.subID, return_counts=True)

    testpersub = [0] * (len(unique_sub) + 1)
    for sub, count in zip(unique_sub, test_count):
        testpersub[sub] = count

    log += f'\nSubID\tnum of test\tnum of incorr\tcorr ratio\n'
    incorr_subs, incorr_counts = np.unique(incorrects_subs, return_counts=True)
    for sub, ic in zip(incorr_subs, incorr_counts):
        log += f'{sub:02d}\t{testpersub[sub]}\t{ic}\t{(testpersub[sub] - ic) / testpersub[sub]:.2f}\n'
    file.write(log)
    print(log)

plot_confusion_matrix(labels, preds, labels_name, path=test_path, lbname=train_name, title=f'{train_name} CM')
print(f'saved in {test_path}')

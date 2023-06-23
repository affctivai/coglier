import os
from os.path import join, exists
from pathlib import Path
import numpy as np
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import DeapDataset_
from utils.model import MyCCNN
from utils.tools import plot_confusion_matrix, get_roc_auc_score

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
random_seed = 42
seed_everything(random_seed)

BATCH = 64
K = 1000
LABEL = '4' # 'v', 'a', '4'
project_name = 'Subdepend'
projcet_path = join(os.getcwd(), 'results', project_name)

def evaluate_sub(SUB):
    DATAS = join(os.getcwd(),"SEED_IV", 'SEED_npz')
    DATA = join(DATAS, 'SubDepen', SUB)
    DNAME = 'seg_DE'
    NAME = f'{DNAME}_{LABEL}'

    if LABEL == 'a': train_name = 'arousal'
    elif LABEL == 'v': train_name = 'valence'
    else: train_name = 'emotion'

    train_path = Path(join(os.getcwd(), 'results', project_name, SUB, train_name))
    # train_path = get_folder(train_path)
    train_path.mkdir(parents=True, exist_ok=True)

    # Load train, valid
    trainset = DeapDataset_(DATA, NAME, 'train')
    validset = DeapDataset_(DATA, NAME, 'valid')

    labels_name = validset.label.tolist()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'trainset: {trainset.x.shape} \t validset: {validset.x.shape}')


    model = MyCCNN(in_channels=trainset.x.shape[1], num_classes=len(labels_name))
    model = model.to(device)

    model.load_state_dict(torch.load(join(train_path, 'best.pt')))

    testset = DeapDataset_(DATA, NAME, 'test')
    print(f'testset: {testset.x.shape}')
    testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)

    def evaluate_test(model, loader, criterion, device):
        losss, accs = [], []
        labels, preds, msps = [], [], []
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
                msps.append(y_pred.max(1)[0].cpu())
                preds.append(y_pred.argmax(1).cpu())
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        msps = torch.cat(msps, dim=0)
        return np.mean(losss), np.mean(accs), labels, preds, msps

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    with open(join(train_path, 'test.txt'), 'w') as file:
        test_loss, test_acc, labels, preds, msps  = evaluate_test(model, testloader, criterion, device)

        msps = msps.numpy()
        top = np.argsort(msps)[::-1]
        top = top[:K]
        precision = preds[list(top)].eq(labels[list(top)]).sum() / K

        log = f"'test_loss':{test_loss:.3f},'test_acc':{test_acc*100:.2f},"
        log += f"'roc_auc_score':{get_roc_auc_score(labels, preds)},"
        log += f"'precision@{K}':{precision:.3f}"
        # log += '\n'+classification_report(labels, preds)
        file.write(log)
        # file.write(f'{LABEL}:{labels_name}\t BATCH {BATCH}\n{DNAME}  testset:{tuple(testset.x.shape)}\n')
        print(log)

    plot_confusion_matrix(labels, preds, labels_name, path=train_path, lbname=train_name, title=f'Subject{SUB} {train_name}')
    print(f'saved in {train_path}')



def save_results(sublist):
    test_results = dict()
    for sub in sublist:
        evaluate_sub(sub)
        file = open(join(projcet_path, sub, train_name, 'test.txt'), 'r')
        result = '{'+ file.readline() + '}'
        test_results[int(sub)] = eval(result)

    df = pd.DataFrame.from_dict(test_results, orient='index')
    df.to_excel(join(projcet_path, f'{train_name}_results2.xlsx'))


lb = '4'
if lb == 'a': train_name = 'arousal'
elif lb == 'v': train_name = 'valence'
else: train_name = 'emotion'

SUBLIST = [str(i) for i in range(1, 16)]

save_results(SUBLIST)
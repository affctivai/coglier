import os
from os.path import exists
import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

def getFromnpz_(dir, sub, out=True, cla='v'):
    sub += '.npz'
    if out: print(sub, end=' ')
    data = np.load(os.path.join(dir, sub), allow_pickle=True)
    datas = data['x']
    if cla == '4': targets = data['y']
    if cla == 'v': targets = data['v']
    if cla == 'a': targets = data['a']
    return datas, targets

def getDataset(path, names, mode):
    path = os.path.join(path, f'{names}_{mode}.npz')
    data = np.load(path, allow_pickle=True)
    datas, targets = data['X'], data['Y']
    return datas, targets

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_roc_auc_score(labels, preds):
    labels_oh = labels.view(-1,1)
    preds_oh = preds.view(-1,1)

    ohe = OneHotEncoder()

    labels_oh = ohe.fit_transform(labels_oh)
    labels_oh = labels_oh.toarray()

    preds_oh = ohe.fit_transform(preds_oh)
    preds_oh = preds_oh.toarray()

    score = f'{roc_auc_score(labels_oh, preds_oh)*100:.2f}'
    return score

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_folder(path):
    if path.exists():
        for n in range(2, 100):
            p = f'{path}{n}'
            if not exists(p):
                break
        path = Path(p)
    return path

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def print_auroc(true, pred):
    fpr, tpr, _ = roc_curve(true, pred, pos_label=1)
    log = f"AUROC: {auc(fpr, tpr)}\n"
    return log

def get_roc_auc_score(labels, preds):
    labels_oh = labels.view(-1,1)
    preds_oh = preds.view(-1,1)

    ohe = OneHotEncoder()

    labels_oh = ohe.fit_transform(labels_oh)
    labels_oh = labels_oh.toarray()

    preds_oh = ohe.fit_transform(preds_oh)
    preds_oh = preds_oh.toarray()

    score = f'{roc_auc_score(labels_oh, preds_oh)*100:.2f}'
    return score
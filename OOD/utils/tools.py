import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
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

def plot_VA(vals, count_v, aros, count_a, path=os.getcwd()):
    plt.figure(figsize=(10, 6))
    w = 0.2
    plt.bar(vals+1 -0.1, count_v, width=w, label='valence')
    plt.bar(aros+1 +0.1, count_a, width=w, label='arousal')

    for i, x in enumerate(vals):
        plt.text(x+1, count_v[i], str(count_v[i]),
                 fontsize=11, color="blue", horizontalalignment='right', verticalalignment='bottom')
        plt.text(x+1, count_a[i], str(count_a[i]),
                 fontsize=11, color="red", horizontalalignment='left', verticalalignment='bottom')

    plt.title('Count per class', fontsize=15)
    plt.ylabel('count', fontsize=12)
    plt.xlabel('Rating', fontsize=12)
    plt.xticks(vals+1, fontsize=10)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(path, dpi=200)


# scheduler plot
class MyScheduler:
    def __init__(self, step, steps, flag='CWR', baselr=0.001, T_0_num= 1, T_up_num= 3, T_mult_num=1):
        self.model = torch.nn.Linear(1,1)
        if flag == 'Exp':
            self.op = optim.Adam(self.model.parameters(), lr=baselr)
            self.sche = lr_scheduler.ExponentialLR(self.op, gamma=0.98)
        elif flag == 'Lam':
            self.op = optim.Adam(self.model.parameters(), lr=baselr)
            self.sche = lr_scheduler.LambdaLR(self.op, lr_lambda=self.func)
        elif flag == 'CWR':
            self.op = optim.Adam(self.model.parameters(), lr=0)
            self.sche = lr_scheduler.CosineAnnealingWarmUpRestarts(self.op, T_0=steps//T_0_num, T_mult=T_mult_num, 
                                                                   eta_max=baselr, T_up=step*T_up_num, gamma=0.5)
        self.step = step
        self.steps = steps

    def make_lrs(self):
        lrs = []
        for step in range(self.steps):
            self.op.step()
            lrs.append(self.op.param_groups[0]['lr'])
            self.sche.step()
        return lrs

    def func(self, epoch):
        if epoch < 30:
            return 0.5 ** 0
        elif epoch < 60:
            return 0.5 ** 1
        elif epoch < 90:
            return 0.5 ** 2
        else: return 0.5 ** 3

def plot_scheduler(lrs, save=False, path=os.getcwd()):
    plt.figure(figsize=(12, 6))
    plt.plot(lrs)
    plt.xlabel('steps', fontsize=15)
    plt.ylabel('Learning Rate', fontsize=15) 
    plt.grid(True)
    if not save: plt.show()
    else: plt.savefig(os.path.join(path,'scheduler.png'), dpi=200)

#------------------------------------------train-------------------------------------------------
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def plot_train_result(train_losses, valid_losses, train_accs, valid_accs, epochs, size=(15,6), path=os.getcwd()):
    fig, loss_ax = plt.subplots(figsize=size)
    acc_ax = loss_ax.twinx()

    xran = range(1, len(train_losses)+1)
    loss_ax.plot(xran, train_losses, 'y', label = 'train loss')
    loss_ax.plot(xran, valid_losses, 'r', label = 'val loss')

    acc_ax.plot(xran, train_accs, 'b', label = 'train accuracy')
    acc_ax.plot(xran, valid_accs, 'g', label = 'valid accuracy')

    loss_ax.set_xlabel('epoch', fontsize=15)
    loss_ax.set_ylabel('loss',fontsize=15)
    acc_ax.set_ylabel('accuracy',fontsize=15)

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    # plt.xticks(range(1, epochs+1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path,'train_result.png'), dpi=200)

#------------------------------------------test-------------------------------------------------
# confusion matrix
def plot_confusion_matrix(labels, preds, label_name, path, lbname, title):
    cm = confusion_matrix(labels, preds, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=label_name)
    fig = plt.figure(figsize = (7, 7))
    ax = fig.add_subplot(1, 1, 1)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)  # delete colorbar
    plt.xlabel(f'Predicted {lbname}', fontsize = 12)
    plt.ylabel(f'True {lbname}', fontsize = 12)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'CM.png'), dpi=200)

# ROC curve
def print_auroc(true, pred, percent=0.95, path=os.getcwd()):
    fpr, tpr, thresholds = roc_curve(true, pred, pos_label=1)
    log = f"AUROC: {auc(fpr, tpr)}\n"
    return_threshold = 0.
    for idx, i in enumerate(tpr):
        if i >= percent:
            # print(idx)
            log += f'FPR at TPR {percent}: {fpr[idx]}\n'
            log += f'Threshold at TPR {percent}: {thresholds[idx]}\n'
            return_threshold = thresholds[idx]
            # print(return_threshold)
            break
    # Plot the ROC curve
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'ROC_curve.png'), dpi=200)
    return log

# Precision-Recall curve
def print_pr(true, pred, dpi=150):
    precision, recall, _ = precision_recall_curve(true, pred)
    print(f'PR AUC: {auc(recall, precision)*100:.1f}')

    # Plot PR curve
    plt.figure(dpi=dpi)
    plt.plot(recall, precision, marker='.')
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

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
#-------------------------------------------external----------------------------------------------

# earlystopping
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

#------------------------------------------- Utility ----------------------------------------------

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
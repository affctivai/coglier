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
from utils.dataset import PreprocessedDataset
from utils.model import MyCCNN, TSCeption, EEGNet
from utils.tools import MyScheduler, plot_scheduler, epoch_time, plot_train_result
from utils.tools import plot_confusion_matrix, get_roc_auc_score
from sklearn.metrics import classification_report

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
# # LABEL = 'v' # 4, v, a
# EPOCH = 200
# BATCH = 64


# ---- DEAP
# DATASET_NAME = "DEAP"
# DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")
# LABEL = 'v' # 4, v, a
# EPOCH = 1
# BATCH = 64

# ---- SEED_IV
DATASET_NAME = "SEED_IV"
DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")
LABEL = '4' # 4, v, a
EPOCH = 100
BATCH = 128

# ---- SEED
# DATASET_NAME = "SEED"
# DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")
# LABEL = '4' # 4, v, a
# EPOCH = 1
# BATCH = 128

# project_name = 'baseline_TSC'
project_name = 'baseline_EEGNet'
# project_name = 'baseline_de'

DATA = join(DATAS, project_name)
DNAME = 'seg'
NAME = f'{DNAME}_{LABEL}'
if LABEL == 'a': train_name = 'arousal'
elif LABEL == 'v': train_name = 'valence'
else: train_name = 'emotion'

# save result in train_path
train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, project_name, train_name))
train_path = get_folder(train_path)

def run_train():
    train_path.mkdir(parents=True, exist_ok=True)

    # Load train, valid
    trainset = PreprocessedDataset(DATA, NAME, 'train')
    validset = PreprocessedDataset(DATA, NAME, 'valid')
    print(f'trainset: {trainset.x.shape} \t validset: {validset.x.shape}')

    labels_name = np.unique(validset.y) + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    # model = MyCCNN(in_channels=trainset.x.shape[1], num_classes=len(labels_name))
    # model = TSCeption(num_electrodes=trainset.x.shape[2], num_classes=len(labels_name))
    model = EEGNet(num_electrodes=trainset.x.shape[2], num_classes=len(labels_name), chunk_size=256)

    model = model.to(device)
    print(summary(model, trainset.x.shape[1:]))
    # return

    trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)

    STEP = len(trainloader)
    STEPS = EPOCH * STEP

    ## plot_scheduler
    # scheduler = MyScheduler(STEP, STEPS, flag='CWR', baselr=0.0001, T_0_num= 1, T_up_num= 3)
    # plot_scheduler(scheduler.make_lrs(), save=True, path=train_path)
    #--------------------------------------train-------------------------------------------------------
    def train(model, loader, optimizer, criterion, scheduler, device, scaler):
        epoch_loss = 0; epoch_acc = 0
        model.train()
        for (x, y, subID) in loader:
            x = x.to(device)
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

            lrs.append(optimizer.param_groups[0]['lr'])
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

    optimizer = optim.Adam(model.parameters(), lr=0, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer,T_0=STEPS,T_mult=1,eta_max=1e-3,T_up=STEP*3,gamma=0.5)

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
                torch.save(model.state_dict(), join(train_path,'best.pt'))

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            log = f'{epoch+1:02} {epoch_secs:2d}s \t {train_loss:1.3f}\t{train_acc*100:6.2f}%\t{valid_loss:1.3f}\t{valid_acc*100:6.2f}%'
            file.write(log + '\n')
            print(log)

    plot_scheduler(lrs, save=True, path=train_path)
    plot_train_result(train_losses, valid_losses, train_accs, valid_accs, EPOCH, size=(9, 5), path=train_path)
    print(f"model weights saved in '{join(train_path,'best.pt')}'")


# run_train()
#--------------------------------------test-------------------------------------------------------
def run_test():
    test_path.mkdir(parents=True, exist_ok=True)

    # Load test
    testset = PreprocessedDataset(DATA, NAME, 'test')
    print(f'testset: {testset.x.shape}')

    labels_name = np.unique(testset.y) + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model (load parameters)
    # model = MyCCNN(in_channels=testset.x.shape[1], num_classes=len(labels_name))
    # model = TSCeption(num_electrodes=testset.x.shape[2], num_classes=len(labels_name))
    model = EEGNet(num_electrodes=testset.x.shape[2], num_classes=len(labels_name))

    model = model.to(device)
    model.load_state_dict(torch.load(join(train_path, 'best.pt')))

    BATCH = 64
    testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)

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

                y_prob = nn.functional.softmax(y_pred, dim=-1)
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



train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, project_name, train_name))
run_train()

test_path = Path(join(train_path), 'test')
test_path = get_folder(test_path)
# run_test()
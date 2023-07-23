import os
from os.path import join, exists
import time
import random
from pathlib import Path
import numpy as np
import argparse

import torch
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.dataset import PreprocessedDataset
from utils.model import CCNN, TSCeption, EEGNet, DGCNN
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

parser = argparse.ArgumentParser()
parser.add_argument("--model", dest="model", action="store", default="CCNN") # CCNN, TSC, EEGNet, DGCNN
parser.add_argument("--label", dest="label", action="store", default="v") # 4, v, a
parser.add_argument("--batch", dest="batch", action="store", default="64") # 64, 128
parser.add_argument("--feature", dest="feature", action="store", default="DE") # DE, PSD
parser.add_argument("--dataset", dest="dataset", action="store", default="GAMEEMO") # GAMEEMO, SEED, SEED_IV, DEAP
parser.add_argument("--epoch", dest="epoch", action="store", default="1") # 1, 50, 100
parser.add_argument("--dropout", dest="dropout", action="store", default="0") # 1, 50, 100

args = parser.parse_args()

DATASET_NAME = args.dataset
LABEL = args.label
MODEL_NAME = args.model
FEATURE = args.feature
BATCH = int(args.batch)
EPOCH = int(args.epoch)
DROPOUT = float(args.dropout)

PROJECT = 'baseline'

if DATASET_NAME == 'GAMEEMO':
    DATAS = join("C:\\", "Users", "LAPTOP", "jupydir", "DATAS", 'GAMEEMO_npz', 'Projects')
    # LABEL = 'v'     # 4, v, a
    # PROJECT = 'baseline'
    # MODEL_NAME = 'DGCNN'    # 'CCNN', 'TSC', 'EEGNet', 'DGCNN'
    # FEATURE = 'PSD'          # 'DE', 'PSD'
    # BATCH = 64
elif DATASET_NAME == 'SEED':
    DATAS = join(os.getcwd(),"datasets", DATASET_NAME, "npz", "Projects")
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


#-----------------------------------------------------------------------------------------
def set_args(project, model_name, feature, label): # 0.1 make dataset과 호환맞춘다면 편의대로...
    if model_name == 'CCNN':
        project_data = '_'.join([project, feature, 'grid'])
        project_name = '_'.join([project, model_name, feature])

    elif model_name in ['TSC', 'EEGNet']:
        project_data = '_'.join([project, 'raw'])
        project_name = '_'.join([project, model_name])

    elif model_name == 'DGCNN':
        project_data = '_'.join([project, feature])
        project_name = '_'.join([project, model_name, feature])

    if label == 'a':    train_name = 'arousal'
    elif label == 'v':  train_name = 'valence'
    else:               train_name = 'emotion'

    data_dir = join(DATAS, project_data)
    data_name = f'{LABEL}'
    return data_dir, data_name, project_name, train_name

DATA, NAME, project_name, train_name = set_args(PROJECT, MODEL_NAME, FEATURE, LABEL)

#--------------------------------------train-------------------------------------------------------
def run_train(model_name):
    train_path.mkdir(parents=True, exist_ok=True)

    # Load train, valid
    trainset = PreprocessedDataset(DATA, NAME, 'train')
    validset = PreprocessedDataset(DATA, NAME, 'valid')
    print(f'trainset: {trainset.x.shape} \t validset: {validset.x.shape}')

    trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
    validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)

    labels_name = np.unique(validset.y) + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    # hyperparameter 최적화 필요...
    if model_name == 'CCNN':
        model = CCNN(num_classes=len(labels_name), dropout=DROPOUT)
        max_lr = 1e-4
        # EPOCH = 100
    elif model_name == 'TSC':
        model = TSCeption(num_electrodes=trainset.x.shape[2], num_classes=len(labels_name), sampling_rate=128, dropout=DROPOUT)
        max_lr = 1e-3
        # EPOCH = 200
    elif model_name == 'EEGNet':
        model = EEGNet(chunk_size=trainset.x.shape[3], num_electrodes=trainset.x.shape[2], num_classes=len(labels_name), dropout=DROPOUT)
        max_lr = 1e-3
        # EPOCH = 200
    elif model_name == 'DGCNN':
        model = DGCNN(in_channels=trainset.x.shape[2], num_electrodes=trainset.x.shape[1], num_classes=len(labels_name))
        max_lr = 1e-3
        # EPOCH = 200

    model = model.to(device)
    print(summary(model, trainset.x.shape[1:]))

    STEP = len(trainloader)
    STEPS = EPOCH * STEP

    optimizer = optim.Adam(model.parameters(), lr=0, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=STEPS, T_mult=1, eta_max=max_lr, T_up=STEP*3, gamma=0.5)

    ## plot_scheduler
    # scheduler = MyScheduler(STEP, STEPS, flag='CWR', baselr=0.0001, T_0_num= 1, T_up_num= 3)
    # plot_scheduler(scheduler.make_lrs(), save=True, path=train_path)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

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
        return epoch_loss / len(loader), epoch_acc / len(loader)

    lrs = []
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    best_valid_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()
    # ----------------------------------------run-------------------------------------------------------
    with open(join(train_path, 'train.txt'), 'w') as file:
        file.write(f'LABEL {LABEL}:{labels_name}\t BATCH {BATCH}\
                    \n{NAME}  train:{tuple(trainset.x.shape)}\tvalid:{tuple(validset.x.shape)}\
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

#--------------------------------------test--------------------------------------------------------
def run_test(model_name):
    test_path.mkdir(parents=True, exist_ok=True)

    # Load test
    testset = PreprocessedDataset(DATA, NAME, 'test')
    print(f'testset: {testset.x.shape}')

    testloader = DataLoader(testset, batch_size=BATCH, shuffle=False)

    labels_name = np.unique(testset.y) + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model (load parameters)
    if model_name == 'CCNN':
        model = CCNN(num_classes=len(labels_name))
    elif model_name == 'TSC':
        model = TSCeption(num_electrodes=testset.x.shape[2], num_classes=len(labels_name), sampling_rate=128)
    elif model_name == 'EEGNet':
        model = EEGNet(chunk_size=testset.x.shape[3], num_electrodes=testset.x.shape[2], num_classes=len(labels_name))
    elif model_name == 'DGCNN':
        model = DGCNN(in_channels=testset.x.shape[2], num_electrodes=testset.x.shape[1], num_classes=len(labels_name))
    model = model.to(device)
    model.load_state_dict(torch.load(join(train_path, 'best.pt')))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    def evaluate_test(model, loader, criterion, device):
        losss, accs = [], []
        labels, preds = [], []
        probs, subIDs = [], []
        model.eval()
        with torch.no_grad():
            for (x, y, subID) in loader:
                x = x.to(device);   y = y.to(device)

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

    # ----------------------------------------run-------------------------------------------------------
    with open(join(test_path, 'output.txt'), 'w') as file:
        file.write(f'{LABEL}:{labels_name}\t BATCH {BATCH}\
                    \n{NAME}  test:{tuple(testset.x.shape)}\n')

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

#---------------------------------------main-------------------------------------------------------
# save result in train_path
train_path = Path(join(os.getcwd(), 'results', DATASET_NAME, project_name, train_name))
train_path = get_folder(train_path)

test_path = Path(join(train_path), 'test')
test_path = get_folder(test_path)

run_train(MODEL_NAME)
run_test(MODEL_NAME)
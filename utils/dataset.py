from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset

# for subject independent--------------------------------------------------------------
def load_list_subjects(src, mode, sublist, label):
    datas, targets = [], []
    for subnum in sublist:
        subnum += '.npz'
        data = np.load(join(src, mode, f'{label}_{subnum}'), allow_pickle=True)
        datas.extend(data['X'])
        targets.extend(data['Y'])
    datas = np.array(datas); targets = np.array(targets)
    return datas, targets

def load_list_subjects_rp(src, mode, sublist, label, z=2.5):
    datas, targets = [], []
    for subnum in sublist:
        subnum += f'_rp_{int(z*100)}.npz'
        data = np.load(join(src, mode, f'{label}_{subnum}'), allow_pickle=True)
        datas.extend(data['X'])
        targets.extend(data['Y'])
    datas = np.array(datas); targets = np.array(targets)
    return datas, targets

class PreprocessedDataset(Dataset):
    def __init__(self, x, Y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(Y[:, 0], dtype=torch.int64)
        self.subID = Y[:, 1]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.subID[idx]

    def __len__(self):
        return self.y.shape[0]

# for subject dependent---------------------------------------------------------------
def load_per_subject(src, mode, subnum, label):
    subnum += '.npz'
    data = np.load(join(src, mode, f'{label}_{subnum}'), allow_pickle=True)
    datas = data['X']
    targets = data['Y']
    return datas, targets

class PreprocessedDataset_(Dataset):
    def __init__(self, x, Y):
        self.x = torch.tensor(x, dtype=torch.float32)
        tmp_y = torch.tensor(Y[:, 0], dtype=torch.int64)
        self.label, self.y = torch.unique(tmp_y, sorted=True, return_inverse=True)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.y.shape[0]
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset

class GameemoDataset(Dataset):
    def __init__(self, src, names, mode):
        self.src = src
        self.names = names
        self.mode = mode

        self.path = join(self.src, f'{self.names}_{self.mode}.npz')

        data = np.load(self.path, allow_pickle=True)
        X, Y = data['X'], data['Y']
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y[:, 0], dtype=torch.int64)
        self.subID = Y[:, 1]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.subID[idx]

    def __len__(self):
        return self.y.shape[0]

# for subject dependent
class GameemoDataset_(Dataset):
    def __init__(self, src, names, mode):
        self.src = src
        self.names = names
        self.mode = mode

        self.path = join(self.src, f'{self.names}_{self.mode}.npz')
        data = np.load(self.path, allow_pickle=True)
        X, Y = data['X'], data['Y']
        self.x = torch.tensor(X, dtype=torch.float32)
        tmp_y = torch.tensor(Y[:, 0], dtype=torch.int64)
        self.label, self.y = torch.unique(tmp_y, sorted=True, return_inverse=True)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.y.shape[0]
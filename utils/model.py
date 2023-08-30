import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# LSTM
Book: Zhang X, Yao L. Deep Learning for EEG-Based Brain-Computer Interfaces: Representations, Algorithms and Applications[M]. 2021.
URL: https://www.worldscientific.com/worldscibooks/10.1142/q0282#t=aboutBook
Related Project: https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/pythonscripts/4-1-1_LSTM.py

# CCNN
Paper: Yang Y, Wu Q, Fu Y, et al. Continuous convolutional neural network with 3D input for EEG-based emotion recognition[C]//International Conference on Neural Information Processing. Springer, Cham, 2018: 433-443.
URL: https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
Related Project: https://github.com/ynulonger/DE_CNN

# TSCeption
Paper: Ding Y, Robinson N, Zhang S, et al. Tsception: Capturing temporal dynamics and spatial asymmetry from EEG for emotion recognition[J]. arXiv preprint arXiv:2104.02935, 2021.
URL: https://arxiv.org/abs/2104.02935
Related Project: https://github.com/yi-ding-cs/TSception

# EEGNet 
Paper: Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
URL: https://arxiv.org/abs/1611.08024
Related Project: https://github.com/braindecode/braindecode/tree/master/braindecode

# DGCNN
Paper: Song T, Zheng W, Song P, et al. EEG emotion recognition using dynamical graph convolutional neural networks[J]. IEEE Transactions on Affective Computing, 2018, 11(3): 532-541.
URL: https://ieeexplore.ieee.org/abstract/document/8320798
Related Project: https://github.com/xueyunlong12589/DGCNN
'''

def get_model_with_dropout(model_name, data_x_shape, num_classes, device, dropout):
    if model_name == 'CCNN':
        model = CCNN(num_classes=num_classes, dropout=dropout)
        max_lr = 1e-4
        return model.to(device), max_lr
    elif model_name == 'TSC':
        model = TSCeption(num_electrodes=data_x_shape[2], num_classes=num_classes, sampling_rate=128, dropout=dropout)
        max_lr = 1e-3
        return model.to(device), max_lr
    elif model_name == 'EEGNet':
        model = EEGNet(chunk_size=data_x_shape[3], num_electrodes=data_x_shape[2], num_classes=num_classes, dropout=dropout)
        max_lr = 1e-3
        return model.to(device), max_lr
    elif model_name == 'DGCNN':
        model = DGCNN(in_channels=data_x_shape[2], num_electrodes=data_x_shape[1], num_classes=num_classes)
        max_lr = 1e-3
        return model.to(device), max_lr
    else:
        print("Unknown Model.")
        exit(1)

def get_model(model_name, data_x_shape, num_classes, device):
    return get_model_with_dropout(model_name, data_x_shape, num_classes, device, 0.5)
    

# ------------------------------------------LSTM----------------------------------------------
class LSTM(nn.Module):
    def __init__(self, num_electrodes: int = 32, num_classes: int = 2,
                 hid_dim = 128, n_layers = 2, dropout = 0.3, bidirectional = False):
        super(LSTM, self).__init__()

        self.num_electrodes = num_electrodes
        self.num_classes = num_classes
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size = num_electrodes, hidden_size = hid_dim, num_layers = n_layers,
                            dropout = dropout, bias = True, # default
                            batch_first = True, bidirectional = bidirectional)

        self.fc = nn.Linear(hid_dim, num_classes)
        
    def forward(self, x):
        out, (h, c) = self.lstm(x, None)   
        out = F.dropout(out, 0.3)
        x = self.fc(out[:, -1, :])  # 마지막 time step만 가지고 
        return x

# ------------------------------------------CCNN-----------------------------------------------
class CCNN(nn.Module): # input_size: batch x freq x 9 x 9
    def __init__(self, in_channels = 4, grid_size = (9, 9), num_classes = 2, dropout = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(self.in_channels, 64, kernel_size=4, stride=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(64, 128, kernel_size=4, stride=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(128, 256, kernel_size=4, stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(256, 64, kernel_size=4, stride=1), nn.ReLU())

        self.lin1 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * 64, 1024),
            nn.SELU(), #)  Not mentioned in paper
            nn.Dropout(self.dropout)) # error Dropout2d
        self.lin2 = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x

# -----------------------------------------TSCeption--------------------------------------------
class TSCeption(nn.Module): # input_size: batch x 1 x EEG channel x datapoint
    def __init__(self, num_electrodes = 28, num_T = 15, num_S = 15, in_channels = 1, hid_channels = 32,
                 num_classes = 2, sampling_rate = 128, dropout = 0.5):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.num_T = num_T
        self.num_S = num_S
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.dropout = dropout

        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(num_electrodes), 1), 1, int(self.pool * 0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(num_electrodes * 0.5), 1), (int(num_electrodes * 0.5), 1),
                                         int(self.pool * 0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(nn.Linear(num_S, hid_channels), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels, num_classes))

    def conv_block(self, in_channels, out_channels, kernel, stride, pool_kernel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(), nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)))

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out

# ------------------------------------------EEGNet----------------------------------------------
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class EEGNet(nn.Module): # input_size: batch x 1 x EEG channel x datapoint
    def __init__(self, chunk_size: int = 151, num_electrodes: int = 60, num_classes: int = 2,
                 F1: int = 8, F2: int = 16, D: int = 2,
                 kernel_1: int = 64, kernel_2: int = 16, dropout: float = 0.5):
        super(EEGNet, self).__init__()
        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.num_classes = num_classes
        self.F1 = F1;   self.F2 = F2;   self.D = D
        self.kernel_1 = kernel_1;   self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1, stride=1, padding=(0, 0), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1*self.D, self.F1*self.D, (1, self.kernel_2),
                      stride=1, padding=(0, self.kernel_2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1*self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False)

    @property   # 메서드를 속성처럼 접근 가능하게 해줌
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)
        return x

# ------------------------------------------DGCNN----------------------------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None: return out + self.bias
        else: return out

class Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)

def normalize_A(A: torch.Tensor, symmetry: bool=False) -> torch.Tensor:
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L

def generate_cheby_adj(A: torch.Tensor, num_layers: int) -> torch.Tensor:
    support = []
    for i in range(num_layers):
        if i == 0:
            support.append(torch.eye(A.shape[1]).to(A.device))
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support

class Chebynet(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, out_channels: int):
        super(Chebynet, self).__init__()
        self.num_layers = num_layers
        self.gc1 = nn.ModuleList()
        for i in range(num_layers):
            self.gc1.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        adj = generate_cheby_adj(L, self.num_layers)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

class DGCNN(nn.Module): # input_size: batch x EEG channel x freq
    def __init__(self, in_channels: int = 5, num_electrodes: int = 14, num_layers: int = 2,
                 hid_channels: int = 32, num_classes: int = 2):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer1 = Chebynet(in_channels, num_layers, hid_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc1 = Linear(num_electrodes * hid_channels, 64)
        self.fc2 = Linear(64, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes))
        nn.init.xavier_normal_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result = self.fc2(result)
        return result

# ---------------------------------------------------------------------------------------
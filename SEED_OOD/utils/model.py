import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------LSTM--------------------------------------------
class MyLSTM(nn.Module):
    def __init__(self, num_electrodes, out_dim,
                 hid_dim = 128, n_layers = 2, dropout_rate = 0.3, bidirectional = False):

        super().__init__()
        
        self.lstm = nn.LSTM(input_size = num_electrodes, 
                            hidden_size = hid_dim, 
                            num_layers = n_layers,
                            dropout = dropout_rate,
                            bias = True,         # default
                            batch_first = True,
                            bidirectional = bidirectional)

        self.fc = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x):
        out, (h, c) = self.lstm(x, None)   
        out = F.dropout(out, 0.3)
        x = self.fc(out[:, -1, :])  # 마지막 time step만 가지고 
        return x

class MyLSTM_DE(nn.Module):
    def __init__(self, inputsize, out_dim,
                 hid_dim = 128, n_layers = 2, dropout_rate = 0.3, bidirectional = False):

        super().__init__()
        
        self.lstm = nn.LSTM(input_size = inputsize, 
                            hidden_size = hid_dim, 
                            num_layers = n_layers,
                            dropout = dropout_rate,
                            bias = True,         # default
                            batch_first = True,
                            bidirectional = bidirectional)

        self.fc = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x):
        out, (h, c) = self.lstm(x, None)   
        out = F.dropout(out, 0.3)
        x = self.fc(out[:, -1, :])
        return x


# ------------------------------------------CNN-----------------------------------------------------
class MyCCNN(nn.Module):
    def __init__(self, in_channels=4, grid_size= (9, 9), num_classes= 2, dropout= 0.5):
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
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = self.conv2(x)
        print(x.size())
        x = self.conv3(x)
        print(x.size())
        x = self.conv4(x)
        print(x.size())
        AAA

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
    



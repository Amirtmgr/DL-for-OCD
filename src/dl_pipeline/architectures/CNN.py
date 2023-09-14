import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes, kernel_size, dropout, activation="relu"):
        super(CNNModel, self).__init__()
        
        print("CNNModel Init: ", input_channels, num_classes, kernel_size, dropout, activation)

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=kernel_size)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=kernel_size)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=kernel_size)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size)
        
        self.fc1 = nn.Linear(256*6, 128) 
        #self.fc1 = nn.Linear(256*3, 128)
        #self.fc1 = nn.Linear(256*11, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x.float())
        #x = self.conv1(x.double())
        #print(x.dtype)
        x = self.batchnorm1(x)
        x = self.act_fn(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act_fn(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.act_fn(x)

        #print(x.shape)

        x = x.view(x.size(0), -1)

        #print(x.shape)
        #print(x.dtype)
        
        x = self.fc1(x)
        x = self.act_fn(x)

        x = self.dropout(x)

        x = self.fc2(x)
        #print(x.shape)
        return x

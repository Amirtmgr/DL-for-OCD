import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes, window_size=150, dropout=0.4):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.3)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * (window_size // 8), 512)
        self.batchnorm_fc = nn.BatchNorm1d(512)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.batchnorm_fc(x)
        x = torch.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

# ------------------------------------------------------------------------
# Description: Implementation of the DeepConvLSTM DL architecture.
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import torch
import torch.nn as nn

#TODO: Sanity check

class DeepConvLSTM(nn.Module):
    """DeepConvLSTM Class.
    
    This architecture combines Conv1D layers and LSTM layers to analyse time series data.
    
    Args:
        config (dict): Dictionary containing the configuration parameters.
            - input_dim (int): Number of input channels.
            - conv_hidden_dim (int): Hidden dimension of the convolutional layers.
            - conv_kernel_size (int): Size of the convolutional kernel.
            - channels (int): Number of sensor channels         
            - lstm_hidden_dim (int): Hidden dimension of the LSTM layers.
            - lstm_layers (int): Number of LSTM layers.
            - lstm_dropout (float): Dropout rate of the LSTM layers.
            - bidirectional (bool): If True, LSTM layers will be bidirectional.
            - classes (int): Number of output classes.
    """
    
    class DeepConvLSTM(nn.Module):
        def __init__(self, config):
            super(DeepConvLSTM, self).__init__()
            
            self.input_dim = config['input_dim'] 
            self.conv_hidden_dim = config['conv_hidden_dim'] 
            self.conv_kernel_size = config['conv_kernel_size'] 
            self.channels = config['channels']
            self.lstm_hidden_dim = config['lstm_hidden_dim'] 
            self.lstm_layers = config['lstm_layers'] 
            self.lstm_dropout = config['lstm_dropout'] 
            self.bidirectional = config['bidirectional'] 
            self.classes = config['classes']

            # TODO: Add batch normalization
            # TODO: Add max pooling
            
            # Convolutional layers
            self.conv1 = nn.Conv1d(self.input_dim, self.conv_hidden_dim, kernel_size=self.conv_kernel_size)
            self.conv2 = nn.Conv1d(self.conv_hidden_dim, self.conv_hidden_dim, kernel_size=self.conv_kernel_size)
            self.conv3 = nn.Conv1d(self.conv_hidden_dim, self.conv_hidden_dim, kernel_size=self.conv_kernel_size)
            self.conv4 = nn.Conv1d(self.conv_hidden_dim, self.conv_hidden_dim, kernel_size=self.conv_kernel_size)
            
            # LSTM layer
            self.lstm = nn.LSTM(self.channels * self.conv_hidden_dim, self.lstm_hidden_dim, num_layers=self.lstm_layers, dropout=self.lstm_dropout, bidirectional=self.bidirectional)
            
            # Fully connected layer
            self.fc = nn.Linear(self.lstm_hidden_dim, self.classes)

            # Activation functions
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # Add dimension to input tensor
            x = x.permute(0, 2, 1) # (batch_size, seq_len, channels)

            # Apply convolutional layers with activation functions
            
            # TODO: Add batch normalization
            # TODO: Add max pooling

            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            
            # Reshape the tensor for LSTM input
            x = x.permute(0, 2, 1)

            # Apply LSTM layer
            x, _= self.lstm(x)
            x = x[:, -1, :] # Get last output of LSTM layer

            # Reshape the tensor for fully connected layer input
            x = x.view(-1, self.lstm_hidden_dim)
            print("x.shape: ", x.shape)

            # TODO: Add dropout layer

            # Apply fully connected layer
            x = self.fc(x)
            return x
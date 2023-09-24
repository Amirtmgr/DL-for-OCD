import torch
import torch.nn as nn
import numpy as np
from src.helper.logger import Logger

class DeepConvLSTM(nn.Module):

    def __init__(self, config):
        
        super(DeepConvLSTM, self).__init__()
    
        self.window_size = config.get('window_size', 150)
        self.num_classes = config.get('num_classes', 3)
        self.sensors = config.get('sensors', 'both')
        
        # CNN Parameters
        self.input_channels = config.get('input_channels', 1)
        self.hidden_channels = []
        self.hidden_channels = config.get('hidden_channels', [64, 64, 64])
        self.kernel_sizes = config.get('kernel_sizes')
        self.cnn_bias = config.get('cnn_bias', False)

        act = config["activation"]["name"]

        # Activation function
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'leaky_relu':
            self.activation = nn.LeakyReLU(config["activation"]["negative_slope"])
        elif act == 'elu':
            self.activation = nn.ELU(config["activation"]["alpha"])
        elif act == 'selu':
            self.activation = nn.SELU()
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            Logger.warn(f"Activation function {act} not found. Using ReLU.")
            self.activation = nn.ReLU()

        Logger.info(f"Input channels: {self.input_channels}")
        Logger.info(f"Hidden channels: {self.hidden_channels}")
        Logger.info(f"Kernel sizes: {self.kernel_sizes}")
        Logger.info(f"CNN bias: {self.cnn_bias}")

        # CNN layers
        self.features = nn.ModuleList()

        # Add convolutional layers
        for idx, channel in enumerate(self.hidden_channels):
            self.features.append(
                nn.Conv2d(self.input_channels, 
                        channel,
                        bias=self.cnn_bias, 
                        kernel_size=(self.kernel_sizes[idx], 1))
                )
            self.features.append(nn.BatchNorm2d(channel))
            self.features.append(self.activation)
            # Update number of channels
            self.input_channels = channel
        
        # LSTM Layers
        self.input_features = 6 if self.sensors == 'both' else 3
        self.lstm_hidden_size = config.get('lstm_hidden_size', 128)
        self.lstm_num_layers = config.get('lstm_num_layers', 2)
        self.lstm_bias = config.get('lstm_bias', False)
        self.lstm_dropout = config.get('lstm_dropout', 0.25) if self.lstm_num_layers > 1 else 0.0
        self.lstm_bidirectional = config.get('lstm_bidirectional', False)
        self.lstm_directions = 2 if self.lstm_bidirectional else 1

        self.lstm = nn.LSTM(
            input_size = self.input_features * self.hidden_channels[-1],
            hidden_size = self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            bias=self.lstm_bias,
            dropout=self.lstm_dropout,
            bidirectional=self.lstm_bidirectional,
            )

        # Dropout Layer
        self.drop_probability = config.get('dropout', 0.25)
        self.dropout = nn.Dropout(self.drop_probability)

        # Fully connected layers
        self.fc_hidden_size = config.get('fc_hidden_size', 128)
        self.output_neurons = self.num_classes if self.num_classes > 2 else 1

        self.fc_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * self.lstm_directions, self.fc_hidden_size),
            nn.BatchNorm1d(self.fc_hidden_size),  # Batch normalization
            self.activation,
            nn.Linear(self.fc_hidden_size, self.output_neurons),
        )

    def forward(self, x):
        batch = x.shape[0]
        # Reshape the tensor for CNN input [batch, channels, window_size, input_features]
        x = x.view(-1, 1, self.window_size, self.input_features)
        
        # Feed forward through CNN layers
        for layer in self.features:
            x = layer(x)

        #print(x.shape)

        # Reshape the tensor for LSTM input
        x = x.permute(0, 2, 1, 3)
        sequence_length = x.shape[1]
        
        x = x.reshape(batch, sequence_length, -1)
        
        #print(x.shape)

        # Feed forward LSTM layer
        x, _  = self.lstm(x)
        #print(x.shape)

        x = x.view(-1, self.lstm_hidden_size * self.lstm_directions)
        #print(x.shape)
        
        # Apply dropout
        x = self.dropout(x)

        # Feed Forward FC layers
        output = self.fc_layers(x)
        
        # Reshape the output
        output = output.view(batch, -1, self.output_neurons)
        output = output[:, -1, :]
        #print(output.shape)
        return output

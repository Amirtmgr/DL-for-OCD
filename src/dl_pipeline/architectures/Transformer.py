import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from src.helper.logger import Logger

class CNNTransformer(nn.Module):

    def __init__(self, config):
        super(CNNTransformer, self).__init__()

        self.window_size = config.get('window_size', 150)
        self.num_classes = config.get('num_classes', 3)
        self.sensors = config.get('sensors', 'both')
        
        # CNN Parameters
        self.input_channels = config.get('input_channels', 1)
        self.hidden_channels = []
        self.hidden_channels = config.get('hidden_channels', [64, 64, 64])
        self.kernel_sizes = config.get('kernel_sizes')
        self.cnn_bias = config.get('cnn_bias', False)
        self.cnn_batch_norm = config.get('cnn_batch_norm', True)

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
        elif act == 'gelu':
            self.activation = nn.GELU()
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
                nn.Conv1d(self.input_channels, 
                        channel,
                        bias=self.cnn_bias, 
                        kernel_size=self.kernel_sizes[idx])
                )
            if self.cnn_batch_norm:
                self.features.append(nn.BatchNorm1d(channel))
            self.features.append(self.activation)
            # Update number of channels
            self.input_channels = channel
        

        # Transformer Layers
        self.input_features = 6 if self.sensors == 'both' else 3
        self.nhead = config.get("multi_attn_heads", 3)
        self.dim_feedforward = config.get("dim_feedforward", 32)
        self.transformer_dropout = config.get("transformer_dropout", 0.25)
        self.transformer_act_fn = config.get("transformer_act_fn", 'relu')
        self.num_encoders = config.get('num_encoder_layers', 2)
        self.num_decoders = config.get('num_decoder_layers', 2)
        self.encode_position = config.get('encode_position', True)
        self.transformer_dim = self.hidden_channels[-1]

        # Encoder Layer
        transformer_encoder_layer = TransformerEncoderLayer(
            d_model = self.transformer_dim,
            nhead = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout = self.transformer_dropout,
            activation = self.transformer_act_fn
        )

        # Transformer encoders
        self.transformer_encoder = TransformerEncoder(
            transformer_encoder_layer,
            num_layers = self.num_encoders,
            norm = nn.LayerNorm(self.transformer_dim)
        )

        # Token
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        # Check Encode position
        if self.encode_position:
            self.positional_embedding =  nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        # Fully connected layers
        self.fc_hidden_size = config.get('fc_hidden_size', 128)
        self.output_neurons = self.num_classes if self.num_classes > 2 else 1
        self.fc_batch_norm = config.get('fc_batch_norm', True)

        # Dropout
        self.drop_probability = config.get('dropout', 0.25)
        self.dropout = nn.Dropout(self.drop_probability)

        # Layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.LayerNorm(self.transformer_dim))
        self.fc_layers.append(nn.Linear(self.transformer_dim, self.fc_hidden_size))
        if self.fc_batch_norm:
            self.fc_layers.append(nn.BatchNorm1d(self.fc_hidden_size))
        self.fc_layers.append(self.activation)
        self.fc_layers.append(self.dropout)
        self.fc_layers.append(nn.Linear(self.fc_hidden_size, self.output_neurons))


        # Init
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    
    def forward(self, x):
        #shape B x L x C = Batch, Window_Size, Num_features

        # Feature Embedding
        x = x.transpose(1,2) 
        
        # Feed forward through CNN layers
        for layer in self.features:
            x = layer(x)

        print(x.shape)

        # Permute for Transformer Layer
        x = x.permute(2, 0, 1)
        print(x.shape)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([cls_token, x])
        print(x.shape)

        # Add position embedding
        if self.encode_position:
            x += self.positional_embedding
        
        print(x.shape)
        # Transformer Encoder
        logits = self.transformer_encoder(x)[0]
        print(logits.shape)
        return logits


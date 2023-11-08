import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from src.helper.logger import Logger
from src.dl_pipeline.architectures.activation import get_act_fn

class CNNTransformer(nn.Module):

    def __init__(self, config):
        super(CNNTransformer, self).__init__()

        self.window_size = config.get('window_size', 150)
        self.num_classes = config.get('num_classes', 3)
        self.sensors = config.get('sensors', 'both')
        self.task_type = config.get('task_type')
        
        # CNN Parameters
        self.input_features = 6 if self.sensors == 'both' else 3
        self.input_channels = self.input_features
        self.hidden_channels = []
        self.hidden_channels = config.get('hidden_channels', [64, 64, 64])
        self.kernel_sizes = config.get('kernel_sizes', [1, 1, 1])
        self.cnn_bias = config.get('cnn_bias', False)
        self.cnn_batch_norm = config.get('cnn_batch_norm', True)

        # Activation function
        self.activation = get_act_fn(config["activation"])

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
                        kernel_size=self.kernel_sizes[idx],
                        padding= self.kernel_sizes[idx]//2)
                )
            if self.cnn_batch_norm:
                self.features.append(nn.BatchNorm1d(channel))
            self.features.append(self.activation)
            # Update number of channels
            self.input_channels = channel
        

        # Transformer Layers
        self.nhead = config.get("multi_attn_heads", 3)
        self.dim_feedforward = config.get("dim_feedforward", 32)
        self.transformer_dropout = config.get("transformer_dropout", 0.25)
        self.transformer_act_fn = config.get("transformer_act_fn", 'relu')
        self.num_encoders = config.get('num_encoder_layers', 2)
        self.num_decoders = config.get('num_decoder_layers', 2)
        self.encode_position = config.get('encode_position', True)
        self.transformer_dim = self.hidden_channels[-1]
        self.transformer_bias = config.get('transformer_bias', False)

        # Encoder Layer
        transformer_encoder_layer = TransformerEncoderLayer(
            d_model = self.transformer_dim,
            nhead = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout = self.transformer_dropout,
            activation = self.transformer_act_fn,
            bias=self.transformer_bias # TODO: Only available in PyTorch >= 2.1.0
        )

        # Transformer encoders
        self.transformer_encoder = TransformerEncoder(
            transformer_encoder_layer,
            num_layers = self.num_encoders,
            norm = nn.LayerNorm(self.transformer_dim)
        )

        # Class Token        
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        # Check Encode position
        if self.encode_position:
            self.positional_embedding = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        # Fully connected layers
        self.fc_hidden_size = config.get('fc_hidden_size', 128)
        self.output_neurons = self.num_classes if self.task_type >= 2 else 1
        self.fc_batch_norm = config.get('fc_batch_norm', True)
        self.fc_bias = config.get('fc_bias', False)

        # Dropout
        self.drop_probability = config.get('dropout', 0.25)
        self.dropout = nn.Dropout(self.drop_probability)

        # Layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.LayerNorm(self.transformer_dim))
        self.fc_layers.append(nn.Linear(self.transformer_dim, self.fc_hidden_size, bias=self.fc_bias))
        if self.fc_batch_norm:
            self.fc_layers.append(nn.BatchNorm1d(self.fc_hidden_size))
        self.fc_layers.append(self.activation)
        self.fc_layers.append(self.dropout)
        self.fc_layers.append(nn.Linear(self.fc_hidden_size, self.output_neurons, bias=self.fc_bias))


        # Init
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    
    def forward(self, x):
        #shape B x L x C = Batch, Window_Size, Num_features
        #print("X: ",x.shape)
        
        x = x.transpose(1,2) # B x C x L
        
        # Feed forward through CNN layers
        for layer in self.features:
            x = layer(x)

        #print("CNN: ",x.shape)
        
        # Permute for Transformer Layer
        x = x.permute(2, 0, 1)
        #print(x.shape)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([cls_token, x])

        # Add position embedding
        if self.encode_position:
            x += self.positional_embedding

        #print("Positional: ",x.shape)

        # Transformer Encoder
        x = self.transformer_encoder(x)[0]

        #print("transformer: ",x.shape)

        # FC layers
        for fc in self.fc_layers:
            x = fc(x)
        
        #print("FC OUT: ", x.shape)
        return x



# Multi-Modal Transformer
class MultiCNNTransformer(nn.Module):

    def __init__(self, config):
        super(MultiCNNTransformer, self).__init__()

        self.window_size = config.get('window_size', 150)
        self.num_classes = config.get('num_classes', 3)
        self.sensors = config.get('sensors', 'both')
        self.task_type = config.get('task_type')
        
        # CNN Parameters
        self.input_features = 6 if self.sensors == 'both' else 3
        self.input_channels = 3 #self.input_features
        self.hidden_channels = []
        self.hidden_channels = config.get('hidden_channels', [64, 64, 64])
        self.kernel_sizes = config.get('kernel_sizes', [1, 1, 1])
        self.cnn_bias = config.get('cnn_bias', False)
        self.cnn_batch_norm = config.get('cnn_batch_norm', True)

        # Activation function
        self.activation = get_act_fn(config["activation"])

        Logger.info(f"Input channels: {self.input_channels}")
        Logger.info(f"Hidden channels: {self.hidden_channels}")
        Logger.info(f"Kernel sizes: {self.kernel_sizes}")
        Logger.info(f"CNN bias: {self.cnn_bias}")

        # CNN layers
        self.features_1 = nn.ModuleList()
        self.features_2 = nn.ModuleList()

        # Add convolutional layers
        for idx, channel in enumerate(self.hidden_channels):
            self.features_1.append(
                nn.Conv1d(self.input_channels, 
                        channel,
                        bias=self.cnn_bias, 
                        kernel_size=self.kernel_sizes[idx],
                        padding= self.kernel_sizes[idx]//2)
                )

            self.features_2.append(
                nn.Conv1d(self.input_channels, 
                        channel,
                        bias=self.cnn_bias, 
                        kernel_size=self.kernel_sizes[idx],
                        padding= self.kernel_sizes[idx]//2)
                )

            if self.cnn_batch_norm:
                self.features_1.append(nn.BatchNorm1d(channel))
                self.features_2.append(nn.BatchNorm1d(channel))
            self.features_1.append(self.activation)
            self.features_2.append(self.activation)

            # Update number of channels
            self.input_channels = channel
        

        # Transformer Layers
        self.nhead = config.get("multi_attn_heads", 3)
        self.dim_feedforward = config.get("dim_feedforward", 32)
        self.transformer_dropout = config.get("transformer_dropout", 0.25)
        self.transformer_act_fn = config.get("transformer_act_fn", 'relu')
        self.num_encoders = config.get('num_encoder_layers', 2)
        self.num_decoders = config.get('num_decoder_layers', 2)
        self.encode_position = config.get('encode_position', True)
        self.transformer_dim = self.hidden_channels[-1]
        self.transformer_bias = config.get('transformer_bias', False)

        # Encoder Layer
        transformer_encoder_layer_1 = TransformerEncoderLayer(
            d_model = self.transformer_dim,
            nhead = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout = self.transformer_dropout,
            activation = self.transformer_act_fn,
            bias=self.transformer_bias # TODO: Only available in PyTorch >= 2.1.0
        )

        transformer_encoder_layer_2 = TransformerEncoderLayer(
            d_model = self.transformer_dim,
            nhead = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout = self.transformer_dropout,
            activation = self.transformer_act_fn,
            bias=self.transformer_bias # TODO: Only available in PyTorch >= 2.1.0
        )

        # Transformer encoders
        self.transformer_encoder_1 = TransformerEncoder(
            transformer_encoder_layer_1,
            num_layers = self.num_encoders,
            norm = nn.LayerNorm(self.transformer_dim)
        )

        self.transformer_encoder_2 = TransformerEncoder(
            transformer_encoder_layer_2,
            num_layers = self.num_encoders,
            norm = nn.LayerNorm(self.transformer_dim)
        )

        # Class Token        
        self.cls_token_1 = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)
        self.cls_token_2 = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        # Check Encode position
        if self.encode_position:
            self.positional_embedding_1 = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))
            self.positional_embedding_2 = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))
        
        # Fully connected layers
        self.fc_hidden_size = config.get('fc_hidden_size', 128)
        self.output_neurons = self.num_classes if self.task_type >= 2 else 1
        self.fc_batch_norm = config.get('fc_batch_norm', True)
        self.fc_bias = config.get('fc_bias', False)

        # Dropout
        self.drop_probability = config.get('dropout', 0.25)
        self.dropout = nn.Dropout(self.drop_probability)

        # Layers
        self.fc_layers = nn.ModuleList()
        new_dim = self.transformer_dim * 2
        #self.layer_norm = nn.LayerNorm(new_dim)
        #self.fc_layer = nn.Linear(new_dim, self.output_neurons, bias=self.fc_bias)

        self.fc_layers.append(nn.LayerNorm(new_dim))
        self.fc_layers.append(nn.Linear(new_dim, self.fc_hidden_size, bias=self.fc_bias))
        if self.fc_batch_norm:
            self.fc_layers.append(nn.BatchNorm1d(self.fc_hidden_size))
        self.fc_layers.append(self.activation)
        self.fc_layers.append(self.dropout)
        self.fc_layers.append(nn.Linear(self.fc_hidden_size, self.output_neurons, bias=self.fc_bias))


        # Init
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    
    def forward(self, x):
        #shape B x L x C = Batch, Window_Size, Num_features
        #print("X: ",x.shape)
        x1 = x[:,:,:3]
        x2 = x[:,:,3:]

        x1 = x1.transpose(1,2) # B x C x L
        x2 = x2.transpose(1,2)

        # Feed forward through CNN layers
        for layer in self.features_1:
            x1 = layer(x1)

        for layer in self.features_2:
            x2 = layer(x2)
        
        #print("CNN: x1 ",x1.shape, "| x2 ", x2.shape)
        
        # Permute for Transformer Layer
        x1 = x1.permute(2, 0, 1)
        x2 = x2.permute(2, 0, 1)

        #print("x1: ",x1.shape)
        #print("x2: ",x2.shape)

        # Prepend class token
        cls_token_1 = self.cls_token_1.unsqueeze(1).repeat(1, x1.shape[1], 1)
        cls_token_2 = self.cls_token_2.unsqueeze(1).repeat(1, x2.shape[1], 1)
        
        #print("cls_token_1: ",cls_token_1.shape)
        #print("cls_token_2: ",cls_token_2.shape)

        x1 = torch.cat([cls_token_1, x1])
        x2 = torch.cat([cls_token_2, x2])

        #print("x1: ",x1.shape)
        #print("x2: ",x2.shape)

        # Add position embedding
        if self.encode_position:
            x1 += self.positional_embedding_1
            x2 += self.positional_embedding_2
            
        #print("Positional: ",x1.shape)
        #print("Positional: ",x2.shape)

        # Transformer Encoder
        x1 = self.transformer_encoder_1(x1)[0]
        x2 = self.transformer_encoder_2(x2)[0]

        #print("transformer: ",x1.shape)
        #print("transformer: ",x2.shape)

        # Concatenate
        x = torch.cat([x1, x2], dim=1)
        #print("Concat: ",x.shape)

        # Element-wise multiplication
        #x = x1 * x2
        #print("Element-wise multiplication: ",x.shape)

        # Element-wise addition
        #x = x1 + x2
        #print("Element-wise addition: ",x.shape)

        # FC layers
        for fc in self.fc_layers:
            x = fc(x)
        
        # x = self.layer_norm(x)
        # x = self.fc_layer(x)
        #print("FC OUT: ", x.shape)
        return x

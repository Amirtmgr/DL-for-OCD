import torch
from torch import nn
from src.helper.logger import Logger

# Activation function
def get_act_fn(act_config):

    # Name of activation function
    act = act_config["name"]

    # Activation function
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(act_config["negative_slope"])
    elif act == 'elu':
        return nn.ELU(act_config["alpha"])
    elif act == 'selu':
        return nn.SELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'gelu':
        return nn.GELU()
    else:
        Logger.warn(f"Activation function {act} not found. Using ReLU.")
        return nn.ReLU()
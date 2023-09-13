import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler, MinMaxScaler, RobustScaler

from src.helper import directory_manager as dm
from src.helper import df_manager as dfm
from src.helper import data_structures as ds
from src.helper.data_model import CSVHeader, HandWashingType
from src.helper.sliding_window import get_windows, process_dataframes

from src.helper.metrics import Metrics
from src.helper.state import State
from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
from src.dl_pipeline.architectures.CNN import CNNModel
from src.dl_pipeline.train import train_model
from src.helper import data_preprocessing as dp

# Function to load network
def load_network():
    # TODO: Check for other networks
    network = cl.config.architecture.name
    num_class = cl.config.architecture.num_classes
    input_channels = cl.config.train.batch_size
    window_size = cl.config.dataset.window_size
    dropout = cl.config.architecture.dropout
    kernel_size = cl.config.architecture.kernel_size
    activation = cl.config.architecture.activation

    if network == "cnn":
        model = CNNModel(window_size,
                         num_class,kernel_size,
                         dropout,
                         activation)
    else:
        return None
    
    return model

def load_optim(model):
    optim_name = cl.config.optim.name
    lr = cl.config.optim.learning_rate
    momentum = cl.config.optim.momentum
    weight_decay = cl.config.optim.weight_decay

    if optim_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,momentum=momentum)


def load_criterion(weights):
    loss = cl.config.criterion.name

    criterion = nn.CrossEntropyLoss()

    if cl.config.criterion.weighted:
        class_weights = weights.to(cl.config.train.device)
        if loss == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print('Applied weighted class weights: ')
            print(class_weights)
    
    return criterion

def load_lr_scheduler(optimizer):
    scheduler = cl.config.lr_scheduler.name
    step_size = cl.config.lr_scheduler.step_size
    gamma = cl.config.lr_scheduler.gamma

    if scheduler == "step_lr":
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return None

def setup_cuda():
    # Check CUDA
    if not torch.cuda.is_available():
        Logger.error("CUDA is not available. Using CPU only.")
        return "cpu"

    if cl.config.train.device == "cuda":
        Logger.info("Using CUDA device.")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        cudnn.fastest = True
        cudnn.deterministic = True
        return device

def train():
    # Setup CUDA
    device = setup_cuda()

    # Load dataset
    train_dataset, val_dataset = dp.get_datasets()

    # Load dataloaders
    train_loader = dp.load_dataloader(train_dataset)
    val_loader = dp.load_dataloader(val_dataset)
    
    # Get class weights
    class_weights = dp.compute_weights(train_dataset)

    # Load Traning parameters
    model = load_network()
    optimizer = load_optim(model)
    criterion = load_criterion(class_weights)
    lr_scheduler = load_lr_scheduler(optimizer)

    # Train Model
    state = train_model(model, criterion, 
                        optimizer, lr_scheduler,
                        train_loader, val_loader, device)

    state.info()

    # Visuals
    state.plot_losses()
    state.plot_f1_scores()
    
    # Inference



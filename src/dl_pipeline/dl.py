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

from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
from src.dl_pipeline.architectures.CNN import CNNModel
from src.dl_pipeline import train as t
from src.helper import data_preprocessing as dp
from src.dl_pipeline import validate as v

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

# Function to train model and return state

def train():
    # Setup CUDA
    device = setup_cuda()

    cv = cl.config.train.cross_validation.name

    if cv == "loso"  or cv == "kfold":
        v.k_fold_cv(device)
    else:
        # Load dataset
        train_dataset, val_dataset = dp.get_datasets()

        # Load dataloaders
        train_loader = dp.load_dataloader(train_dataset)
        val_loader = dp.load_dataloader(val_dataset)
        
        # Get class weights
        class_weights = dp.compute_weights(train_dataset)

        # Load Traning parameters
        model = t.init_weights(t.load_network())
        optimizer = t.load_optim(model)
        criterion = t.load_criterion(class_weights)
        lr_scheduler = t.load_lr_scheduler(optimizer)

        # Train Model
        state = t.train_model(model, criterion, 
                            optimizer, lr_scheduler,
                            train_loader, val_loader, device)

        state.info()

        # Visuals
        state.plot_losses()
        state.plot_f1_scores()
    
    # Inference




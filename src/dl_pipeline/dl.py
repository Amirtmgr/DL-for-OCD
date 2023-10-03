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
from src.helper.data_model import CSVHeader, HandWashingType, LABELS
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
        num_gpus = torch.cuda.device_count()
        Logger.info(f"Number of GPUs: {num_gpus}")
        print("GPUs Count:", num_gpus)
        
        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        cudnn.fastest = True
        cudnn.deterministic = True
        return device

# Function to train model and return state

def train():
    # Setup CUDA
    device = setup_cuda()
    print("Device:", device)
    
    # Check if Multi-GPUs
    #multi_gpu = t.ddp_setup()
    multi_gpu = cl.config.world_size > 1

    cv = cl.config.train.cross_validation.name
    num_classes = cl.config.dataset.num_classes

    if num_classes == 2:
        msg = "==============Binary classification============="
        
        if cl.config.train.cHW_detection:
            msg += "\n=============== rHW vs cHW =============="
            cl.config.dataset.labels = cl.config.dataset.labels[1:]
        else:
            msg += "\n=============== Null vs HW =============="
            cl.config.dataset.labels = ["Null", "HW"]
    elif num_classes == 3:
        msg = "==============Multiclass classification========="
    else:
        raise ValueError("Number of classes must be 2 or 3 in config yaml file")
    
    Logger.info(msg)
    print(msg)
    cl.print_config_dict()

    if cv == "loso"  or cv == "kfold":
        v.subwise_k_fold_cv(device, multi_gpu)
    elif cv == "stratified":
        v.stratified_k_fold_cv(device, multi_gpu)
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
                            train_loader, val_loader, device,
                            threshold=cl.config.train.threshold)

        state.info()

        # Visuals
        state.plot_losses()
        state.plot_f1_scores()
    
    # Clean up
    #t.ddp_destroy()




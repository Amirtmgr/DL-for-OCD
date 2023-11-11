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

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler, MinMaxScaler, RobustScaler

from src.helper import directory_manager as dm
from src.helper import df_manager as dfm
from src.helper import data_structures as ds
from src.helper.data_model import CSVHeader, HandWashingType, LABELS
from src.helper.sliding_window import get_windows, process_dataframes
from src.helper.data_model import TaskType

from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
from src.dl_pipeline.architectures.CNN import CNNModel
from src.dl_pipeline import train as t
from src.helper import data_preprocessing as dp
from src.dl_pipeline import validate as v
from src.dl_pipeline import personalize as p

# Function to setup random seed
def setup_random_seed():
    seed = cl.config.train.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Function to setup CUDA
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

    # Setup random seed
    setup_random_seed()

    # Check if Multi-GPUs
    #multi_gpu = t.ddp_setup()
    
    print("Using", torch.cuda.device_count(), "GPUs!")
    
    multi_gpu = cl.config.world_size > 1

    Logger.info(f"Using {torch.cuda.device_count()} GPUs!")

    cv = cl.config.train.cross_validation.name
    task_type = TaskType(cl.config.dataset.task_type)
    cl.config.train.task_type = task_type
    num_classes = 2 if task_type.value < 2 else 3
    cl.config.dataset.num_classes = num_classes
    cl.config.architecture.num_classes = num_classes

    if task_type == TaskType.rHW_cHW_binary:
        msg = "==============Binary classification============="
        msg += "\n=============== rHW vs cHW =============="
        cl.config.dataset.labels = ["rHW", "cHW"]
    elif task_type == TaskType.cHW_detection:
        msg = "==============Binary classification============="
        msg += "\n=============== Null vs cHW =============="
        cl.config.dataset.labels = ["Null", "cHW"]
    elif task_type == TaskType.HW_detection:
        msg = "==============Binary classification============="
        msg += "\n=============== Null vs HW =============="
        cl.config.dataset.labels = ["Null", "HW"]
    elif task_type == TaskType.HW_classification:
        msg = "==============Multi-class classification============="
        msg += "\n=============== rHW vs cHW =============="
        cl.config.dataset.labels = ["rHW", "cHW"]
    elif task_type == TaskType.Multiclass_classification:
        msg = "==============Multi-class classification========="
        cl.config.dataset.labels = ["Null", "rHW", "cHW"]
    else: 
        raise ValueError("Task type not found in config file.")
    
    Logger.info(msg)
    print(msg)
    cl.print_config_dict()

    if cl.config.dataset.personalization:
        if cl.config.train.ensemble:
            p.ensemble(device, multi_gpu)
        else:
            p.run(device, multi_gpu)
    else: 
        if cv == "loso"  or cv == "kfold":
            v.subwise_k_fold_cv(device, multi_gpu)
        elif cv == "stratified":
            v.stratified_k_fold_cv(device, multi_gpu)
            # TODO: Remove this
            for optim in ['adam', 'sgd']:
                cl.config.optim.name = optim
                for wt in [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0003]
                    cl.config.optim.weight_decay = wt
                    ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
                    for lr in [0.01, 0.001, 0.0001, 0.00005, 0.00001]:
                        cl.config.optim.learning_rate = lr
                        for r in ratios:
                            cl.config.dataset.train_ratio = r
                            p.run(device, multi_gpu)
        elif cv == "personalized":
            p.run(device, multi_gpu)
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




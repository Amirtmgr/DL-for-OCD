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

# Prepare Dataset
def prepare_dataset():
    """Method to prepare dataset
    Returns:
        torch.utils.data.Dataset: Dataset
    """
    #set random seed
    random.seed(cl.config.dataset.random_seed)
   

    if cl.config.dataset.folder == "test" or "features" or "OCD_Export":
        csv_files = dm.get_files_names()
        grouped_files = ds.group_by_subjects(csv_files)
        Logger.info(f"Grouped files for subjects: {grouped_files.keys()}")

        if cl.config.dataset.personalization:
            #Todo: Add personalized subject
            subject = cl.config.dataset.personalized_subject
            Logger.info(f"Personalized subject: {subject}")

        # Train and val dataframes
        train_dfs = []
        val_dfs = []

        subjects = list(grouped_files.keys())
        random.shuffle(subjects)

        if cl.config.debug:
            subjects = subjects[:2]
        # Remove Personalized subject
        try:
            subjects.remove(int(cl.config.dataset.personalized_subject))
        except:
            Logger.info("Personalized subject not found in the subjects list.")
        # Train and val subjects
        train_subjects = subjects
        val_subjects = []

        # Check train-test split
        if cl.config.dataset.train_ratio != 1.0:
            train_size = int(float(cl.config.dataset.train_ratio) * len(subjects))
            train_subjects = subjects[:train_size]
            val_subjects = subjects[train_size:]            

        # Get window_size
        windows_size = cl.config.dataset.window_size
        overlapping_ratio = cl.config.dataset.overlapping_ratio

        # Load all csv files:
        for sub_id in subjects:
            files = grouped_files[sub_id]
            temp_df = dfm.load_all_files(files)

            # To do: Check for windows and labels arrays
            if sub_id in train_subjects:
                train_dfs.append(temp_df)
            elif sub_id in val_subjects:
                val_dfs.append(temp_df)

    
    # Get samples and targets for val and train sets
    train_samples, train_labels = process_dataframes(train_dfs, windows_size, overlapping_ratio)
    val_samples, val_labels = process_dataframes(val_dfs, windows_size, overlapping_ratio)

    # Compute weights
    class_weights = torch.from_numpy(compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)).double()

    # Transform
    #transform = transforms.ToTensor()
    
    train_tensor = torch.from_numpy(train_samples)
    val_tensor = torch.from_numpy(val_samples)
    train_labels_tensor = torch.from_numpy(train_labels)
    val_labels_tensor = torch.from_numpy(val_labels)

    # Create datasets
    train_dataset = TensorDataset(train_tensor,
                                  train_labels_tensor)
    
    if val_samples.size:
        val_dataset = TensorDataset(val_tensor,
                                    val_labels_tensor)
        return train_dataset, val_dataset, class_weights
    
    # Todo: check dimensions
    return train_dataset, None, class_weights
    

def load_dataloaders(train_dataset, val_dataset):
    """Method to load train and val dataloaders
    Args:
        dataset (torch.utils.data.Dataset): Dataset
        batch_size (int): Batch size
        shuffle (bool): Shuffle
        num_workers (int): Number of workers
    Returns:
        torch.utils.data.DataLoader: Dataloader
    """

    # Parse config
    batch_size = cl.config.train.batch_size
    shuffle = cl.config.dataset.shuffle
    num_workers = cl.config.dataset.num_workers
    pin_memory = cl.config.dataset.pin_memory

    gen = torch.Generator()
    gen.manual_seed(cl.config.dataset.random_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, 
                            pin_memory=pin_memory,
                             generator=gen)
    
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, 
                            pin_memory=pin_memory,
                             generator=gen)
        return train_loader, val_loader
    
    return train_loader, None



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
    train_dataset, val_dataset, class_weights = prepare_dataset()

    # Load dataloaders
    train_loader, val_loader = load_dataloaders(train_dataset, val_dataset)
    
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

    # Todos
    # Visuals

    # Inference



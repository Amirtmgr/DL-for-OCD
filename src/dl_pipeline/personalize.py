import datetime
import gc
import numpy as np
import random
from collections import Counter

import torch
from torch.utils.data import ConcatDataset, TensorDataset

from src.dl_pipeline import train as t
from src.helper import data_preprocessing as dp
from src.helper import df_manager as dfm
from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
from src.helper.filters import band_pass_filter
from src.helper import object_manager as om
from src.helper import directory_manager as dm
from src.helper.state import State
from src.helper import data_structures as ds

from imblearn.under_sampling import OneSidedSelection, NearMiss, RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split

def run(device, multi_gpu=False):
    print("Device:", device)

    print("======"*5)
    # start
    start = datetime.datetime.now()
    Logger.info(f"Personalization Start time: {start}")
    print(f"Personalization Start time: {start}")

    is_binary = cl.config.dataset.num_classes < 3
    shelf_name = cl.config.dataset.name
    random_seed = cl.config.dataset.random_seed
    shuffle = cl.config.dataset.shuffle
    cv_name = cl.config.train.cross_validation.name
    personalized_subject = str(cl.config.dataset.personalized_subject)
    train_ratio = cl.config.dataset.train_ratio

    # Load python dataset
    X_dict, y_dict = dp.load_shelves(shelf_name, personalized_subject)

    X_personalized = X_dict[personalized_subject]
    y_personalized = y_dict[personalized_subject]

    Logger.info(f"X_infer shape: {X_personalized.shape} | y_infer shape: {y_personalized.shape}")

    del X_dict, y_dict
    gc.collect()

    # Split data
    if shuffle:
        X_train, X_val, y_train, y_val = train_test_split(X_personalized, y_personalized, train_size = train_ratio, stratify = y_personalized, shuffle=shuffle, random_state = random_seed)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_personalized, y_personalized, train_size = train_ratio)

    Logger.info(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")

    # Create datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).float())
    train_loader = dp.load_dataloader(train_dataset, multi_gpu)
    val_loader = dp.load_dataloader(val_dataset, multi_gpu)
    
    # Load Traning parameters
    model = t.load_network(multi_gpu)
    model = model.to(device)
    print("Model loadded on device: ", device)
    print("Multiple GPUs: ", multi_gpu)


    # Checkpoint
    filename = cl.config.train.checkpoint

    state_checkpoint = t.load_checkpoint(filename)
    print("State: ", state_checkpoint.best_model.state_dict())

    if state_checkpoint is None:
        Logger.info("No checkpoint loaded")

    # Compute weights
    #class_weights = state_checkpoint.best_criterion_weight

    class_weights = dp.compute_weights(y_train)
    class_weights = class_weights.to(device)
    optimizer = t.load_optim(state_checkpoint.best_model, multi_gpu)
    criterion = t.load_criterion(class_weights).to(device)
    lr_scheduler = t.load_lr_scheduler(optimizer)
    
    # Train Model
    state = t.train_model(state_checkpoint.best_model, criterion, 
                        optimizer, lr_scheduler,
                        train_loader, val_loader, device, optional_name=f"_personalized",
                        is_binary=is_binary,
                        threshold= cl.config.train.binary_threshold)
    state.info()
    #state.scalar = scaler

    # Visuals
    state.plot_losses(title=f" Personalized on {personalized_subject} | {cl.config.file_name}")
    state.plot_f1_scores(title=f" Personalized on {personalized_subject} | {cl.config.file_name}")
    
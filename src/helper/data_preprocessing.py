import itertools
import gc
import numpy as np
import os
import shelve
import random

from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
import src.helper.directory_manager as dm
from src.helper import df_manager as dfm
from src.helper import data_structures as ds
from src.helper.sliding_window import process_dataframes, get_windows

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

GROUPED_FILES = None


def get_files():
    """Method to get subjects
    Returns:
        list: Subjects
    """
    csv_files = dm.get_files_names()
    GROUPED_FILES = ds.group_by_subjects(csv_files)

    grouped_files = GROUPED_FILES.copy()

    Logger.info(f"Grouped files for subjects: {grouped_files.keys()}")

    subject = cl.config.dataset.personalized_subject
    Logger.info(f"Personalized subject: {subject}")
           
    if cl.config.dataset.personalization:
        # Return personalized subject and it's files
        return {subject: grouped_files[subject]}
    else:
        #Remove personalized subject
        del grouped_files[subject]

    # Check if debug mode is on
    if cl.config.debug:
        return dict(itertools.islice(grouped_files.items(), 3))
    
    
    # Return all subjects and their files
    return grouped_files


def split_subjects(subjects):
    random.seed(cl.config.dataset.random_seed)
    random.shuffle(subjects)

    # Remove personalized subject
    if cl.config.dataset.personalized_subject in subjects:
        subjects.remove(cl.config.dataset.personalized_subject)
    
    # If Debug
    if cl.config.debug:
        subjects = subjects[:4]

    # For test train split
    if cl.config.dataset.train_ratio != 1.0:
        train_size = int(float(cl.config.dataset.train_ratio) * len(subjects))
        train_subjects = subjects[:train_size]
        inference_subjects = subjects[train_size:]     
        return train_subjects, inference_subjects

def get_datasets():
    """Method to get datasets
    Returns:
        tupple: Train and val datasets
    """
    # Get files
    grouped_files = get_files()
    
    # Get train and val subjects
    train_subjects, val_subjects = split_subjects(grouped_files)
    
    # Get train and val dataframes
    train_df = dfm.load_dfs_from(train_subjects, grouped_files, add_sub_id=True)
    val_df = dfm.load_dfs_from(val_subjects, grouped_files, add_sub_id=True)
    
    # Regex for sensor
    if cl.config.dataset.sensor == "acc":
        regex = "acc*"
        remove_regex = "gyro*"

    elif cl.config.dataset.sensor == "gyro":
        regex = "gyro*"
        remove_regex = "acc*"
    else:
        regex = "acc*|gyro*"
        remove_regex = "none"

    # Scale dataframes
    scaler = get_scaler()
    columns = train_df.filter(regex=regex).columns.tolist()

    train_scaled = scaler.fit_transform(train_df.filter(regex=regex))
    train_df[columns] = train_scaled
    train_df.drop(train_df.filter(regex=remove_regex).columns, axis=1, inplace=True)

    val_scaled = scaler.transform(val_df.filter(regex=regex))
    val_df[columns] = val_scaled
    val_df.drop(val_df.filter(regex=remove_regex).columns, axis=1, inplace=True)

    # Check datasets:
    check_time = not (cl.config.dataset.name == "features")
        
    # Load datasets
    train_windows, train_labels = process_dataframes([train_df], cl.config.dataset.window_size, cl.config.dataset.overlapping_ratio, check_time)
    val_windows, val_labels = process_dataframes([val_df], cl.config.dataset.window_size, cl.config.dataset.overlapping_ratio, check_time)

    # del 
    del train_df, val_df, train_scaled, val_scaled
    gc.collect()

    train_windows_tensor = torch.from_numpy(train_windows)
    train_labels_tensor = torch.from_numpy(train_labels)
    val_windows_tensor = torch.from_numpy(val_windows)
    val_labels_tensor = torch.from_numpy(val_labels)

    # del
    del train_windows, train_labels, val_windows, val_labels
    gc.collect()

    # Create dataset
    train_dataset = TensorDataset(train_windows_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_windows_tensor, val_labels_tensor)

    # del
    del train_windows_tensor, train_labels_tensor, val_windows_tensor, val_labels_tensor
    gc.collect()

    return train_dataset, val_dataset

def create_dataset(df):
    # Check datasets:
    check_time = not (cl.config.dataset.name == "features")
        
    # Load as windows
    windows, labels = process_dataframes([df], cl.config.dataset.window_size, cl.config.dataset.overlapping_ratio, check_time)
    
    # Convert to tensor
    windows_tensor = torch.from_numpy(windows)
    labels_tensor = torch.from_numpy(labels)

    # Create dataset
    dataset = TensorDataset(windows_tensor, labels_tensor)

    del windows,labels, windows_tensor, labels_tensor
    gc.collect()
    return dataset

def load_dataset(dataFrames):

    """Method to load dataset
    Args:
        dataFrames (list): List of dataframes
    Returns:
        torch.utils.data.Dataset: Dataset
    """
    
    # Get window_size
    windows_size = cl.config.dataset.window_size
    overlapping_ratio = cl.config.dataset.overlapping_ratio

    # Get samples and targets for val and train sets
    samples, labels = process_dataframes(dataFrames, windows_size, overlapping_ratio, check_time=True)

    # Convert to torch tensors
    samples_tensor = torch.from_numpy(samples)
    labels_tensor = torch.from_numpy(labels)

    # Create dataset
    dataset = TensorDataset(samples_tensor, labels_tensor)
    
    # del
    del samples, labels, samples_tensor, labels_tensor
    gc.collect()
    
    return dataset
    


# Function to prepare dataloaders
def load_dataloader(dataset):
    """Method to load train and val dataloaders
    Args:
        dataset (torch.utils.data.Dataset): Dataset
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

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, 
                            pin_memory=pin_memory,
                             generator=gen)
    
    return data_loader
  
# Function to compute class weights
def compute_weights(dataset):
    """Method to compute class weights
    Args:
        labels (list): Labels
    Returns:
        list: Class weights
    """

    # Get labels
    if isinstance(dataset, TensorDataset):
        labels = dataset.tensors[1].numpy()
    elif isinstance(dataset, list):
        labels = np.concatenate([d.tensors[1].numpy() for d in dataset])
    else:
        raise TypeError("Dataset must be either TensorDataset or list of TensorDataset.")

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.from_numpy(class_weights).float()


def get_scaler():

    # Parse config
    scaler_type = cl.config.dataset.scaler_type

    # Initialize the scaler based on the selected type
    
    if scaler_type == 'MinMax':
        scaler = MinMaxScaler()
        Logger.info("Using MinMaxScaler.")
    elif scaler_type == 'Robust':
        scaler = RobustScaler()
        Logger.info("Using RobustScaler.")
    elif scaler_type == 'Standard':
        scaler = StandardScaler()
        Logger.info("Using StandardScaler.")
    else:
        Logger.warning("Invalid scaler type. Use 'MinMax', 'Robust', or 'Standard'. Set to default StandardScaler.")
    
    return scaler


def load_shelves(filename):
    # Create path
    path = dm.create_folder("datasets", dm.FolderType.data)
    x_path = os.path.join(path, filename+"_X")
    y_path = os.path.join(path, filename+"_y")
    
    # Open shelves
    X_db = shelve.open(x_path, 'r')
    y_db = shelve.open(y_path, 'r')

    # Convert to dict
    X = dict(X_db)
    y = dict(y_db)

    # Close the db
    X_db.close()
    y_db.close()

    return x, y 
     

import itertools
import gc
import numpy as np
import os
import shelve
import random
from collections import Counter

from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
import src.helper.directory_manager as dm
from src.helper.data_model import TaskType
from src.helper import df_manager as dfm
from src.helper import data_structures as ds
from src.helper.sliding_window import process_dataframes, get_windows

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

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
    personalized_subject = str(cl.config.dataset.personalized_subject)
    
    if personalized_subject in subjects:
        Logger.info(f"Removing personalized subject from dataset...:{personalized_subject}")
        print("Removing personalized subject:")
        subjects.remove(personalized_subject)

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
def load_dataloader(dataset, multi_gpu=False, shuffle=None):
    """Method to load train and val dataloaders
    Args:
        dataset (torch.utils.data.Dataset): Dataset
    Returns:
        torch.utils.data.DataLoader: Dataloader
    """

    # Parse config
    batch_size = cl.config.train.batch_size
    if shuffle is None:
        shuffle = cl.config.dataset.batch_shuffle
    num_workers = cl.config.dataset.num_workers
    pin_memory = cl.config.dataset.pin_memory

    gen = torch.Generator()
    gen.manual_seed(cl.config.dataset.random_seed)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,generator=gen, num_workers=num_workers, pin_memory=pin_memory)
    
    # if multi_gpu:
    #     return DataLoader(dataset, batch_size=batch_size, shuffle=False,
    #                         num_workers=num_workers,
    #                         pin_memory=True, 
    #                          sampler=DistributedSampler(dataset))
    # else:
    #     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
    #                         num_workers=num_workers, pin_memory=pin_memory,
    #                         generator=gen)
    
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
    elif isinstance(dataset, np.ndarray):
        labels = dataset
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


def load_shelves(filename, subjects=None):
    # Filterd subjects
    remove_subjects = cl.config.dataset.filter_subjects
    personalization = cl.config.dataset.personalization
    
    if cl.config.dataset.trustworthy_only:
        print("Using Trustworthy subjects only.")
        subjects = cl.config.dataset.trustworthy_subjects
        print("Trustworthy subjects: subjects")
        Logger.info(f"Using trustworthy subjects only: {subjects}")

    # Create path
    path = dm.create_folder("datasets", dm.FolderType.data)
    x_path = os.path.join(path, filename+"_X")
    y_path = os.path.join(path, filename+"_y")
    
    # Open shelves
    X_db = shelve.open(x_path, 'r')
    y_db = shelve.open(y_path, 'r')

    # Empty dict
    X = {}
    y = {}
    
    if subjects is None:
        # Convert to dict
        subjects = list(X_db.keys())        
    elif isinstance(subjects, str):
        subjects = [subjects]
    

    for subject in subjects:
        if subject in remove_subjects and not personalization:
            print(f"Removing subject: {subject}")
            continue
        X[subject] = X_db[subject]
        y[subject] = y_db[subject]
    # Close the db
    X_db.close()
    y_db.close()

    if cl.config.dataset.num_classes == 2:
        X, y = prepare_binary_data(X, y, cl.config.train.task_type)

    return X, y 
     

def filter_out_class(data, labels, class_name):
    """
    Filter data and labels based on a specified class name.

    Args:
        data (numpy.ndarray): The data array containing samples.
        labels (numpy.ndarray): The labels array containing class labels.
        class_name (int): The name of the class to filter out.

    Returns:
        numpy.ndarray: The filtered data array.
        numpy.ndarray: The filtered labels array.
    """
    # Find the indices of samples belonging to the specified class
    class_indices = np.where(labels != class_name)

    # Use the indices to extract the data and labels for the specified class
    filtered_data = data[class_indices]
    filtered_labels = labels[class_indices]

    return filtered_data, filtered_labels

def replace_labels(labels, label_mapping):
    """
    Replace labels in a numpy array using a mapping dictionary.

    Args:
        labels (numpy.ndarray): The numpy array of labels to be replaced.
        label_mapping (dict): A dictionary mapping old labels to new labels.

    Returns:
        numpy.ndarray: The numpy array with replaced labels.
    """
    # Use numpy's vectorized operations to replace labels
    # If a label is not in the mapping, keep it unchanged
    replaced_labels = np.vectorize(label_mapping.get)(labels)

    return replaced_labels

def prepare_binary_data(X_dict:dict, y_dict:dict, task_type:TaskType):
    """Method to prepare binary data
    Args:
        X_dict (dict): X dictionary
        y_dict (dict): y dictionary
        task_type (TaskType): To prepare data for different types of tasks.
    Returns:
        dict: X dictionary
        dict: y dictionary
    """

    for subject in list(X_dict.keys()):
        X = X_dict[subject]
        y = y_dict[subject]
        
        if task_type == TaskType.HW_classification or task_type == TaskType.rHW_cHW_binary:
            # Filter classes
            temp_X, temp_y = filter_out_class(X, y, 0)
            
            if temp_y.size == 0:
                Logger.warning(f"Subject: {subject} | No rHW data found. Skipping.")
                del X_dict[subject]
                del y_dict[subject]
                Logger.info(f"Removed subject: {subject}. Reason: No rHW/cHW data found.")
                continue
            # Replace labels
            temp_y = replace_labels(temp_y, {1:0, 2:1})

             # Add to dict
            X_dict[subject] = temp_X
                    
        elif task_type == TaskType.HW_detection:
            # Replace labels
            temp_y = replace_labels(y, {0:0, 1:1, 2:1})

        elif task_type == TaskType.cHW_detection:
            # Replace labels
            temp_y = replace_labels(y, {0:0, 1:0, 2:1})
        
        y_dict[subject] = temp_y

        Logger.info(f"For Subject: {subject} | Before: {Counter(y)} | After: {Counter(temp_y)}")
        
    return X_dict, y_dict


def split_data(data, train_ratio, validation_ratio, random_seed=None):
    if not (0 <= train_ratio <= 1) or not (0 <= validation_ratio <= 1) or not (train_ratio + validation_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1 and sum to 1.")
    
    inference_ratio = 1 - train_ratio - validation_ratio

    train_data, temp_data = train_test_split(data, test_size=1 - train_ratio, random_state=random_seed)
    validation_data, inference_data = train_test_split(temp_data, test_size=inference_ratio / (inference_ratio + validation_ratio), random_state=random_seed)
    
    return train_data, validation_data, inference_data

# Normalize the array to the desired range
def scale_arr(arr, bound = None):
    # Calculate the mean and standard deviation along the sensor_channels axis
    channel_means = np.mean(arr, axis=2, keepdims=True)
    channel_stddev = np.std(arr, axis=2, keepdims=True)

    # Normalize the array
    normalized_arr = (arr - channel_means) / channel_stddev

    # Scale the normalized values to the desired range
    min_value = np.min(normalized_arr)
    max_value = np.max(normalized_arr)
    
    normalized_arr = (normalized_arr - min_value) / (max_value - min_value)  # Scale to [0, 1]
    
    if bound:
        lower_bound, upper_bound = bound
        normalized_arr = normalized_arr * (upper_bound - lower_bound) + lower_bound  # Scale to [lower_bound, upper_bound]

    return normalized_arr


# 
def calculate_rms(data):
    return np.sqrt(np.mean(data**2))

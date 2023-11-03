import os
import sys
from enum import Enum, IntEnum
import datetime
from src.utils.config_loader import config_loader as cl

# Types of folders
class FolderType(Enum):
    charts = "saved/charts"
    logs = "saved/logs"
    models = "saved/models"
    results = "saved/results"
    data = "data"
    
# Function to generate name based on date_time or root_date_time
def generate_name(root=""):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{root}_{formatted_datetime}" if root else formatted_datetime
    return file_name

# Function to get working data directory
def get_data_dir():
    # data directory
    data_dir = os.path.join(cl.config.data_path, cl.config.dataset.name)
    return data_dir

# Function to get csv files from folder
def get_files_names(path=None):
    if not path:
        path = get_data_dir()
    
    # Check if exists
    if not os.path.exists(path):
        raise  FileNotFoundError(f"Path does not exists: {path}")
    files = os.listdir(path)
    files_names = [item for item in files if item.endswith(".csv")]
    return sorted(files_names)
        
# Function to create folder and return path
def create_folder(name, folder_type:FolderType ):
    """
    Create folders recursively.

    Args:
        name (str): Name of the folder to create.
    """
    full_path = os.path.join(cl.config.main_path, folder_type.value, name)  
        
    # Create path if doesn't exists
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(full_path, " Directory created.")
    
    return full_path


def get_all_files(path):
    """
    Get all files recursively from a directory.

    Args:
        path (str): Path of the directory.

    Returns:
        list: List of all files.
    """
    
    for root, d, files in os.walk(path):
        print(f"r: {root}, d: {d}, f: {files}")
    return root, sorted(files)


def get_all_models(path):
    """
    Get all best models from a directory.

    Args:
        path (str): Path of the directory.

    Returns:
        list: List of all best models.
    """
    root, files = get_all_files(path)
    
    models = {}

    for file in files:
        fold = file.split("-")[1].split("_")[0]
        if fold not in models.keys():
            models[fold] = []
        models[fold].append(os.path.join(root.split('/')[-1], file))
    
    return models

def get_best_models(path):
    """
    Get all  models from a directory.

    Args:
        path (str): Path of the directory.

    Returns:
        list: List of all best models.
    """
    models = get_all_models(path)
    best_models = {}

    for fold, files in models.items():
        best_models[fold] = sorted(files)[-1]
    
    return best_models

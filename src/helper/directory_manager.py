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
    data_dir = os.path.join(cl.config.data_path, cl.config.dataset.folder)
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




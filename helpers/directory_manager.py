import os
import sys
import datetime

# Function to generate name based on date_time or root_date_time
def generate_name(root=""):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{root}_{formatted_datetime}" if root else formatted_datetime
    return file_name

# Function to get working data directory
def get_data_dir(subfolder = "OCDetect_Export"):
    # current working directory
    current_dir = os.getcwd()
    parent_directory = os.path.dirname(current_dir)
    # data directory
    data_dir = os.path.join(current_dir, "data/"+subfolder)
    return data_dir

# Function to get csv files from folder
def get_files_names(path):
    # Check if exists
    if not os.path.exists(path):
        raise  FileNotFoundError(f"Path does not exists: {path}")
    
    files = os.listdir(path)
    
    files_names = [item for item in files if item.endswith(".csv")]
    
    return files_names
        
# Function to create folder and return path
def create_folder(name):
    """
    Create folders recursively.

    Args:
        name (str): Name of the folder to create.
    """
    
    folder_path = get_data_dir(name)
    
    # Create path if doesn't exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(folder_path, " Directory created.")
    
    return folder_path
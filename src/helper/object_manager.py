import os
import pickle as pk
from src.helper import directory_manager as dm

# Function to save object to file
def save_object(obj, folder, folder_type, file_name):
    
    folder_path = dm.create_folder(folder, folder_type)
    
    file_path = os.path.join(folder_path, file_name)
    
    try:
        with open(file_path, 'wb') as file:
            pk.dump(obj, file)
        return "Object has been successfully saved to file: " + file_path
    except Exception as e:
        return "Error encountered while saving the object: " + str(e)

# Function to load object
def load_object(folder, folder_type, file):
    
    folder_path = dm.create_folder(folder, folder_type)
    file_path = os.path.join(folder_path, file_name)
    
    try:
        with open(file_path, 'rb') as file:
            obj = pk.load(file)
        return obj
    except Exception as e:
        return "Error encountered while loading the object: " + str(e)


def load(path):
    """
    Load file from a path.

    Args:
        path (str): Path of the file.

    Returns:
        list: List of all lines in the file.
    """
    try:
        with open(path, 'rb') as file:
            obj = pk.load(file)
        return obj
    except Exception as e:
        return "Error encountered while loading the object: " + str(e)


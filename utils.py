# ------------------------------------------------------------------------
# Description: Utilies Module
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import os
import pandas as pd
import yaml
from data_models import config

# TODO(atm):Review

def get_config_vars(file_name="config.yml"):
    """Function to get variables from config.yml file.

    Args:
        file_name (str): Full name of the config file
    Returns:
        dictionary
    """
       
    try:
        config_loader = ConfigLoader()
        config_loader.load_config(config_file)
        config_loader.export_to_environment()
    except FileNotFoundError as e:
        print(str(e))

    
    
def get_subjects():
    """Function to loop through subjects' folders.

    Args:
        path (str): The directory path of the datasets
    
    Returns:

    """

    for subject in range (1, 31):
        subject_id = str(folder).zfill(2) # Pad with leading zero    
        for filename in os.listdir(subject_id):
            if filename.endswith(".csv"): # Check if csv file
                file_path = os.path.join(subject_id, filename)
                load_csv(file_path)

def load_subject_CSVs(id:int):
    path = 
    for file in os.listdir()
    
                
def load_CSV(file_path, chunk_size, window_size):
    """Function to load csv file.

    Args:
        file_path (str): The path of the csv file

    Returns:
        True if success, False otherwise

    """

    data = pd.read_csv(file_path)
    window_size = 200
    num_samples = len(data)
    
    for i in range(0, num_samples, window_size):
        window = data.iloc[i:i+window_size]
        plot_window(window)

import src.helper.df_manager as dfm
import src.helper.data_structures as ds
import src.helper.directory_manager as dm
from src.utils.config_loader import config_loader as cl
from src.helper.filters import band_pass_filter
from src.helper.data_preprocessing import create_dataset
from src.helper import sliding_window as sw 

import pandas as pd
import numpy as np
import gc
import shelve
import os

def prepare_datasets(filename):
    # Get all files
    csv_files = dm.get_files_names()
    grouped_files = ds.group_by_subjects(csv_files)
    subjects = list(grouped_files.keys())
    path = dm.create_folder("datasets", dm.FolderType.data)
    x_pth = full_path = os.path.join(path, filename+"_X")
    y_pth = full_path = os.path.join(path, filename+"_y")

    x_shelf = shelve.open(x_pth, 'c')
    y_shelf = shelve.open(y_pth, 'c')
    
    # Loop through each subjects:
    for sub_id in subjects:
        files = grouped_files[sub_id]
        temp_df = dfm.load_all_files(files).drop(['sub_id'], axis=1, errors='ignore')
        # get datasets
        #datasets = create_dataset(temp_df)
        print("Windowing data...")
        print(temp_df.info)
        data, labels = sw.get_windows(temp_df, cl.config.dataset.window_size, overlapping_ratio=cl.config.dataset.overlapping_ratio, check_time=True)
        print(f"{sub_id}: {len(data)}")
        x_shelf[str(sub_id)] = np.array(data, dtype='float32')
        y_shelf[str(sub_id)] = np.array(labels, dtype = 'uint8')
        print(f"Saved {sub_id}")    
    # Close the db
    x_shelf.close()
    y_shelf.close()


def load_dataset(sub_id, filename="OCDetect_datasets"):
    X_pth = os.path.join(dm.create_folder("datasets", dm.FolderType.data), filename + "_X")
    y_pth = os.path.join(dm.create_folder("datasets", dm.FolderType.data), filename + "_y")
    
    X_shelf = shelve.open(X_pth, 'r')
    y_shelf = shelve.open(y_pth, 'r')
    
    X, y  = X_shelf[str(sub_id)], y_shelf[str(sub_id)]
    X_shelf.close()
    y_shelf.close()
    return X, y
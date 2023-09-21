import src.helper.df_manager as dfm
import src.helper.data_structures as ds
import src.helper.directory_manager as dm
from src.utils.config_loader import config_loader as cl
from src.helper.filters import band_pass_filter
from src.helper.data_preprocessing import create_dataset

import pandas as pd
import numpy as np
import gc
import shelve
import os

def prepare_datasets():
    # Get all files
    csv_files = dm.get_files_names()
    grouped_files = ds.group_by_subjects(csv_files)
    subjects = list(grouped_files.keys())
    path = dm.create_folder("datasets", dm.FolderType.data)
    filename = "OCDetect_datasets"
    full_path = os.path.join(path, filename)
    with shelve.open(full_path, 'w' ) as db:

        # Loop through each subjects:
        for sub_id in [20, 21,22,24,25,27,29,30]:
            files = grouped_files[sub_id]
            temp_df = dfm.load_all_files(files).drop(['sub_id'], axis=1, errors='ignore')
            # get datasets
            datasets = create_dataset(temp_df)
            print(f"{sub_id}: {len(datasets)}")
            db[str(sub_id)] = datasets
            print(f"Saved {sub_id}")
        
        # Close the db
        db.close()

def load_dataset(sub_id, filename="OCDetect_datasets"):
    full_path = os.path.join(dm.create_folder("datasets", dm.FolderType.data), filename)
    with shelve.open(full_path, 'r') as db:
        dataset = db[str(sub_id)]
        db.close()
    return dataset

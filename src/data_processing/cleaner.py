import src.helper.df_manager as dfm
import src.helper.data_structures as ds
import src.helper.directory_manager as dm
from src.utils.config_loader import config_loader as cl
from src.helper.filters import band_pass_filter

import pandas as pd
import numpy as np
import gc

def clean_all():
    # Get all files
    csv_files = dm.get_files_names()
    grouped_files = ds.group_by_subjects(csv_files)

    # Loop through each subjects:
    for sub_id in grouped_files.keys():
        files = grouped_files[sub_id]
        temp_df = dfm.load_all_files(files, add_sub_id=True)
        temp_df = clean(temp_df)
        # save
        file_name = f"OCDetect_{sub_id:02}_processed.csv"
        dfm.save_to_csv(temp_df, "processed", file_name)
    
    print("Complete cleaning all files.")

def clean(df):
    print("Cleaning data...")
    print(df.info)
    # Remove rows with ignore flag
    df = dfm.del_ignored_rows(df)

    df = df.drop(columns=['ignore'])

    # Remove rows with NaN
    df.dropna(inplace=True)

    # Bandpass filter parameters
    order = cl.config.filter.order
    fc_high = cl.config.filter.fc_high
    fc_low = cl.config.filter.fc_low
    columns = df.filter(regex='acc*|gyro*').columns.tolist()
    fs = cl.config.filter.sampling_rate    
    
    # Apply Band-pass filter
    df_filtered = band_pass_filter(df, order, fc_high, fc_low, columns, fs)

    # Reset index
    df_filtered.reset_index(drop=True, inplace=True)
 
    # Del df
    del df
    gc.collect()

    return df_filtered

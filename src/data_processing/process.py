import src.helper.df_manager as dfm
import src.helper.data_structures as ds
import src.helper.directory_manager as dm
from src.utils.config_loader import config_loader as cl
from src.helper.filters import band_pass_filter
from src.helper.data_preprocessing import create_dataset
from src.helper import sliding_window as sw 
from src.helper.logger import Logger

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


# Filter
def filter(df):
    order = cl.config.filter.order
    fc_high = cl.config.filter.fc_high
    fc_low = cl.config.filter.fc_low
    columns = df.filter(regex='acc*|gyro*').columns.tolist()
    fs = cl.config.filter.sampling_rate    
    
    Logger.info(f"Filtering data with order: {order}, fc_high: {fc_high}, fc_low: {fc_low}, columns: {columns}, fs: {fs}")
    # Apply Band-pass filter
    df_filtered = band_pass_filter(df, order, fc_high, fc_low, columns, fs)

    # Reset index
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered

# Create subset
def prepare_subset():
    # Get all files
    csv_files = dm.get_files_names()
    
    grouped_files = ds.group_by_subjects(csv_files)
    Logger.info(f"Subjects Grouped files: {grouped_files.keys()}")

    # Loop through each subjects:
    for sub_id in grouped_files.keys():
        # Skip first 12 subjects
        if sub_id <= 12:
            Logger.info(f"Skipping subject: {sub_id}")
            continue

        Logger.info(f"Processing subject: {sub_id}")
        files = grouped_files[sub_id]
        temp_df = dfm.load_all_files(files, add_sub_id=True)
        temp_df = filter(temp_df)
        sub_df = extract_samples(temp_df).copy()
        Logger.info(f"Subset shape: {sub_df.shape}")
        
        counts = sub_df['relabeled'].value_counts()

        Logger.info(f"Subset: {sub_id} | {sub_df['relabeled'].value_counts()}")
        
        if len(counts) == 3:
            if counts[0]/(counts[1]+counts[2]) == 2:
                Logger.info("Found 2:1 ratio")

        # save
        file_name = f"OCDetect_{sub_id:02}_subset.csv"
        dfm.save_to_csv(sub_df, "subset", file_name)
        Logger.info(f"Subset saved: {file_name}")

# Remove consecutive difference
def remove_consecutive_difference(nums, threshold):
    Logger.info(f"Removing consecutive difference with threshold: {threshold}")
    Logger.info(f"Before |  Numbers: {nums}")
    # Initialize an empty result list with the first number from the input list
    result = [nums[0]]

    # Iterate through the input list starting from the second element
    for i in range(1, len(nums)):
        diff = nums[i] - result[-1]
        if diff > threshold:
            result.append(nums[i])
    Logger.info(f"After  |  Numbers: {result}")
    return result

# Extract samples
def extract_samples(df, step_size=1900):
    
    Logger.info(f"Extracting samples with step size: {step_size} and df shape: {df.shape}")
    # Empty list to store selected sample chunks
    selected_chunks = []

    orig_step_size = 1901

    # Row indices with labels 1 and 2
    rows_indices = df[(df['relabeled'] == 1) | (df['relabeled'] == 2)].index[::orig_step_size]
    
    # Remove consecutive indices with difference less than 3 times the step size
    sample_indices = remove_consecutive_difference(rows_indices, step_size*3)

    print(f"Sample indices: {sample_indices} | Length: {len(sample_indices)}")
    Logger.info(f"Sample indices: {sample_indices} | Length: {len(sample_indices)}")
    # Find small chunks of data with labels 1 and 2 with it's preceding and succeeding null samples without ignored types
    for index in sample_indices:
        print( "index: ", index)
        Logger.info(f"Index: {index}")
        lower_index = index - step_size
        upper_index = index + (step_size * 2)

        chunk = df.iloc[lower_index:upper_index, :]
        Logger.info(f"Chunk shape: {chunk.shape}")

        if chunk[chunk['ignore']>0].any().any():
            print("ignore index: ", index)
            Logger.info(f"Ignore index: {index}")
            continue
            
       
        #prev null
        lower_null = index - step_size
        upper_null = index
        
        # Add the previous null chunk to the list
        if lower_null >= 0 and upper_null < len(df):
            prev_null = df.iloc[lower_null:upper_null, :].copy()
            prev_null.loc[:,('relabeled')] = prev_null['relabeled'].mode()[0]
            print("First NUll: ",prev_null.shape)
            selected_chunks.append(prev_null)
            Logger.info(f"prev_null shape: {prev_null.shape}")
        
         #sample
        lower_sample = index
        upper_sample = index + step_size

        # Add sample to the list
        if lower_sample >= 0 and upper_sample < len(df):
            sample = df.iloc[lower_sample:upper_sample, :].copy()
            sample.loc[:,('relabeled')] = sample['relabeled'].mode()[0]
            print("Sample: ",sample.iloc[0]['relabeled'],sample.shape)
            selected_chunks.append(sample)
            Logger.info(f"sample shape: {sample.shape}")

        #next null
        lower_next_null = upper_sample
        upper_next_null = lower_next_null + step_size

        # Add the next null chunk to the list
        if lower_next_null >= 0 and upper_next_null < len(df):
            next_null = df.iloc[lower_next_null:upper_next_null, :].copy()
            next_null.loc[:,('relabeled')] = next_null['relabeled'].mode()[0]
            print("Next NUll: ",next_null.shape)
            selected_chunks.append(next_null)
            Logger.info(f"next_null shape: {next_null.shape}")


    Logger.info(f"Selected chunks: {len(selected_chunks)}")
        
    # combine the selected chunks into a single DataFrame
    new_df = pd.concat(selected_chunks)

    #if df.duplicated(['datetime']).any():
        #print("Duplicated rows exist in the DataFrame.")
        # remove duplicated rows based on 'datetime'
        #new_df.drop_duplicates(subset='datetime', inplace=True)

    # Reset the index
    new_df.reset_index(drop=True, inplace=True)
    
    Logger.info(f"New df shape: {new_df.shape}")

    return new_df
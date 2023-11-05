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
from collections import Counter

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


def load_dataset(sub_id, filename="OCDetect_datasets", with_date = False):
    X_pth = os.path.join(dm.create_folder("datasets", dm.FolderType.data), filename + "_X")
    y_pth = os.path.join(dm.create_folder("datasets", dm.FolderType.data), filename + "_y")
    z_pth = os.path.join(dm.create_folder("datasets", dm.FolderType.data), filename + "_z")

    X_shelf = shelve.open(X_pth, 'r')
    y_shelf = shelve.open(y_pth, 'r')

    X, y  = X_shelf[str(sub_id)], y_shelf[str(sub_id)]
    X_shelf.close()
    y_shelf.close()
    
    if with_date:
        z_shelf = shelve.open(z_pth, 'r')
        z = z_shelf[str(sub_id)]
        z_shelf.close()
        return X, y, z
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


def make_datasets(filename):
    # Get all files
    csv_files = dm.get_files_names()
    grouped_files = ds.group_by_subjects(csv_files)
    subjects = list(grouped_files.keys())
    path = dm.create_folder("datasets", dm.FolderType.data)
    x_pth = os.path.join(path, filename+"_X")
    y_pth = os.path.join(path, filename+"_y")
    z_pth = os.path.join(path, filename+"_z")
    sensors = 6

    x_shelf = shelve.open(x_pth, 'c')
    y_shelf = shelve.open(y_pth, 'c')
    z_shelf = shelve.open(z_pth, 'c')

    # Loop through each subjects:
    for sub_id in subjects:
        print(f"Processing subject: {sub_id}")

        files = grouped_files[sub_id]
        temp_df = dfm.load_all_files(files).drop(['sub_id'], axis=1, errors='ignore')
        # get datasets
        #datasets = create_dataset(temp_df)

        # Remove ignore rows
        temp_df = dfm.del_ignored_rows(temp_df)
        temp_df = temp_df.drop(columns=['ignore'])

        # Remove rows with NaN
        temp_df.dropna(inplace=True)

        # Separate labels

        df_null = temp_df[temp_df["relabeled"] == 0].copy()
        df_rHW = temp_df[temp_df["relabeled"] == 1].copy()
        df_cHW = temp_df[temp_df["relabeled"] == 2].copy()
        
        df_null.reset_index(drop=True, inplace=True)
        df_rHW.reset_index(drop=True, inplace=True)
        df_cHW.reset_index(drop=True, inplace=True)
        
        print("Null labels: \n", {df_null.info()})
        print("rHW labels: \n", {df_rHW.info()})
        print("cHW labels: \n", {df_cHW.info()})
        
        del temp_df
        gc.collect()

        print("Windowing data...")
        
        windows_null, labels_null, datetimes_null = sw.get_windows(df_null, cl.config.dataset.window_size, overlapping_ratio=cl.config.dataset.overlapping_ratio, check_time=True, keep_date=True)
        #windows_rHW, labels_rHW, datetimes_rHW = sw.get_windows(df_rHW, cl.config.dataset.window_size, overlapping_ratio=cl.config.dataset.overlapping_ratio, check_time=True, keep_date=True)
        #windows_cHW, labels_cHW, datetimes_cHW = sw.get_windows(df_cHW, cl.config.dataset.window_size, overlapping_ratio=cl.config.dataset.overlapping_ratio, check_time=True, keep_date=True)

        # Evenly windwing rHW and cHW
        windows_rHW, labels_rHW, datetimes_rHW = window_events(df_rHW)
        windows_cHW, labels_cHW, datetimes_cHW = window_events(df_cHW)

        print("Null labels: \n", {len(windows_null)})
        print("rHW labels: \n", {len(windows_rHW)})
        print("cHW labels: \n", {len(windows_cHW)})

        del df_null, df_rHW, df_cHW
        gc.collect()
        
        # Convert to numpy
        np_windows_null = np.array(windows_null, dtype='float32').reshape(-1, cl.config.dataset.window_size * sensors)
        np_labels_null = np.array(labels_null, dtype = 'uint8')
        np_datetimes_null = np.array(datetimes_null, dtype = 'datetime64[ns]')
        np_windows_rHW = np.array(windows_rHW, dtype='float32').reshape(-1, cl.config.dataset.window_size * sensors)
        np_labels_rHW = np.array(labels_rHW, dtype = 'uint8')
        np_datetimes_rHW = np.array(datetimes_rHW, dtype = 'datetime64[ns]')
        np_windows_cHW = np.array(windows_cHW, dtype='float32').reshape(-1, cl.config.dataset.window_size * sensors)
        np_labels_cHW = np.array(labels_cHW, dtype = 'uint8')
        np_datetimes_cHW = np.array(datetimes_cHW, dtype = 'datetime64[ns]')

        del windows_null, labels_null, datetimes_null, windows_rHW, labels_rHW, datetimes_rHW, windows_cHW, labels_cHW, datetimes_cHW
        gc.collect()

        # Create single dataset
        windows = np.concatenate((np_windows_null, np_windows_rHW, np_windows_cHW), axis=0)
        labels = np.concatenate((np_labels_null, np_labels_rHW, np_labels_cHW), axis=0)
        datetimes = np.concatenate((np_datetimes_null, np_datetimes_rHW, np_datetimes_cHW), axis=0)

        # Sort by datetime
        sorted_order = np.argsort(datetimes)
        sorted_windows = windows[sorted_order]
        sorted_labels = labels[sorted_order]
        sorted_datetimes = datetimes[sorted_order]

        del windows, labels, datetimes
        gc.collect()

        # Save 
        x_shelf[str(sub_id)] = sorted_windows.reshape(-1, cl.config.dataset.window_size, sensors)
        y_shelf[str(sub_id)] = sorted_labels
        z_shelf[str(sub_id)] = sorted_datetimes

        print(f"{sub_id}: {len(sorted_windows)}")
        print(f"Label distribution: {Counter(sorted_labels)}")
        print(f"Saved {sub_id}")

    # Close the db
    x_shelf.close()
    y_shelf.close()
    z_shelf.close()


# Window events with event length of 1901
    def window_events(df, event_length=1901):

        all_windows = []
        all_labels = []
        all_datetimes = []

        start = 0
        end = event_length

        while end <= len(df):
            # Get event instances removing last row to window evenly
            temp_df = df.iloc[start:end, :].copy()
            temp_df.reset_index(drop=True, inplace=True)
            windows, labels, datetimes = sw.get_windows(temp_df, cl.config.dataset.window_size, overlapping_ratio=cl.config.dataset.overlapping_ratio, check_time=False, keep_date=True)
            all_windows.append(windows)
            all_labels.append(labels)
            all_datetimes.append(datetimes)
            start += event_length
            end += event_length
        
        # Flatten
    np_windows = np.concatenate(all_windows, axis=0)
    np_labels = np.concatenate(all_labels, axis=0)
    np_datetimes = np.concatenate(all_datetimes, axis=0)

    del all_windows, all_labels, all_datetimes
    gc.collect()
    
    print(f"Windowed: {len(np_windows)}")
    print(f"Label distribution: {Counter(np_labels)}")

    return np_windows, np_labels, np_datetimes

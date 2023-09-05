import pandas as pd
import numpy as np
from src.helper.logger import Logger


def append_window(df, window_size:int):
    """
    Checks for ignored rows and return last index
    
    Returns:
    
    A boolean indicating whether to proceed, and the index of the last row to consider.
    """
    # Get all rows having ignored label
    df_temp = df.loc[(df['ignore'] > 0)]
    
    # Get time diff
    df['datetime'] = pd.to_datetime(df['datetime'])
    time_diff = df['datetime'].diff()
    time = pd.Timedelta(milliseconds=20)
    indices = time_diff.index[(time_diff > time) | (time_diff < time)]
    
    # Case I: High time difference, top priority
    if not indices.empty:
        #print("High time differences: ", indices)
        return False, indices[0]
    
    # Case II: Ignored rows
    elif len(df_temp) > 0:
        #print("Ignored rows: ", df_temp.index[-1]+1)
        return False, df_temp.index[-1]+1
    else:
        return True, df.index[-1]+1
        

def get_windows(df, window_size:int, overlapping_ratio:float, check_time=False, keep_id=False):
    """
    Returns a list of windows with window_size and overlap_ratio.
    
    Args:
    df: The dataframe to get windows from.
    window_size: The size of each window.
    overlapping: The amount of overlapping ratio between windows.
    Returns:
    A list of windows and labels.
    """
    columns = df.columns.tolist()
    overlapping = int(window_size * overlapping_ratio)
    windows = []
    labels = []
    start = 0
    end = window_size
    window_id = 0
    
    if overlapping >= window_size:
        Logger.error(f"Error: Overlapping ratio should be lesser than 1.0.")
        raise Error("Overlapping ratio should be lesser than 1.0.")
    
    Logger.info(f"Windowing dataframe at window size of {window_size} and overlapping ratio of {overlapping_ratio}:")
    while end <= df.shape[0]:
        temp_df = df.iloc[start:end, :].copy()
        
        if check_time: 
            append, start = append_window(temp_df, window_size)
        else:
            append = True
            start += window_size
            
        if append:
            #print(f"Appended:{window_id}")
            
            # Label window based on majority voting
            temp_df["relabeled"] = temp_df['relabeled'].mode()[0]
            
            if keep_id:
                # Append unique id 
                temp_df["window_id"] = window_id
            
            # Remove ignore column
            if 'ignore' in columns:
                temp_df.drop(['ignore'], axis=1, inplace=True)
            
            # Append window to list
            windows.append(temp_df.drop(["relabeled"], axis=1))
            
            # Append one label per window
            labels.append(temp_df["relabeled"].iloc[0])

            # Reindexing
            start -= overlapping
            window_id += 1
            
        end = start + window_size
    Logger.info(f"Total windows:{window_id}")
    return windows, labels
    

# Function to process a list of DataFrames and return input and target data
def process_dataframes(dataframes, window_size, overlapping_ratio):
    """
    Returns input and target data from a list of DataFrames.

    Args:
    dataframes: A list of DataFrames.
    window_size: The size of each window.
    overlapping: The amount of overlapping ratio between windows.
    Returns:
    A tuple of input and target data.
    """

    # Initialize empty lists for samples and labels
    all_samples = []
    all_labels = []
    
    # Process each DataFrame in the list
    for df in dataframes:
        samples, labels = get_windows(df, window_size, overlapping_ratio)
        all_samples.extend(np.array(samples, dtype='float64'))
        all_labels.extend(np.array(labels, dtype='int64'))
    
    # Convert lists to numpy arrays
    input_data = np.array(all_samples, dtype='float64')
    target_data = np.array(all_labels, dtype='int64')
    
    return input_data, target_data
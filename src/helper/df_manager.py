import os
import re
import gc
import pandas as pd
from tqdm import tqdm

import src.helper.directory_manager as dm
from src.helper.data_model import CSVHeader,HandWashingType
from src.helper.sampler import sample
from src.helper.logger import Logger as log

from src.utils.config_loader import config_loader as cl

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# Function to get params:
def read_params():
    if cl.config.dataset.folder == "OCDetect_Export":
        usecols=['datetime','acc x','acc y','acc z','gyro x','gyro y','gyro z','ignore','relabeled']
        dtype={'relabeled': 'uint8','ignore':'uint8'}
        parse_dates = ["datetime"]
        return usecols, dtype, parse_dates
    else:
        dtype={'relabeled': 'uint8'}
        return None, dtype, None


# Read csv file
def read_csv_file(filename, chunksize=None):
    # Path
    path = os.path.join(dm.get_data_dir(), filename)
    
    # Params
    usecols, dtype, parse_dates = read_params()
    
    try:
        df = pd.read_csv(path, chunksize=chunksize, usecols=usecols,parse_dates=parse_dates, dtype=dtype)
        return df
    
    except FileNotFoundError:
        log.error(f"Error: File '{filename}' not found.")
    except pd.errors.EmptyDataError:
        log.error(f"Error: File '{filename}' is empty.")    
    except pd.errors.ParserError:
        log.error(f"Error: File '{filename}' is corrupted or has parsing issues.")
    
    return pd.DataFrame()

# Clean data
def filter_ignored_rows(df_data:pd.DataFrame):
    if CSVHeader.IGNORE.value in df_data.columns.tolist():
        filtered_df = df_data[df_data[CSVHeader.IGNORE.value] <= 0]
        return filtered_df
    return df_data

# Get iterator of csv file reader.
def get_iterator(filename, chunksize=150):
    path = os.path.join(dm.get_data_dir(),filename)
    
    try:
        df = pd.read_csv(filename, path)
        #log.info("CSV file is readable.")
        return df
    except FileNotFoundError:
        log.error(f"Error: File '{filename}' not found.")
        return None
    except pd.errors.EmptyDataError:
        log.error(f"Error: File '{filename}' is empty.")
        return None
    except pd.errors.ParserError:
        log.error(f"Error: File '{filename}' is corrupted or has parsing issues.")
        return None

# Function to get the csv files of the selected subjects
def filter_by(subject:int, files:list):
    
    selected_files = []
    
    # Regex pattern
    regex_pattern = r"OCDetect_(\d+)_"
    
    log.info(f"Searching regex pattern: {regex_pattern}")
    
    for file in files:
        match = re.search(regex_pattern, file)
        if match and int(match.group(1)) == subject:
            selected_files.append(file)

    return selected_files

# Function to print all rows of csv file
def print_rows(df_sensor, chunk=1000):
    for i in range(0, len(df_sensor), chunk):
        print(df_sensor.iloc[i:i+chunk])
        
# Functiont to calculate mean and var and quartiles
def save_to_csv(df_data, folder, file_name):
    
    folder_path = dm.create_folder(cl.folder, dm.FolderType.data)
    
    # Output File path
    file_path = os.path.join(folder_path, file_name.rsplit( ".", 1 )[0] + f"-{folder}.csv")
    
    log.info(f"Saving CSV File: {file_path}")
    
    # Save the updated DataFrame to a new CSV file
    try:
        df_data.to_csv(file_path, sep=",", index=False)
        log.info(f"DataFrame saved successfully to path: {file_path}.")
    except Exception as e:
        msg = "Error encountered while saving the dataframe: " + str(e)
        log.critical(msg)
        raise Exception(msg)
    
    # release memory
    del df_data
    gc.collect()

# Function to get new header names from dataframe having rows and columns headers
def get_new_cols_header(df, column_first=True):
    
    # Safety check
    if df.empty:
        log.warning("DataFrame is empty.")
        return
    
    # Get rows and cols headers
    row = df.index if column_first else df.columns
    cols = df.columns if column_first else df.index
    
    new_header = []
    
    # Loop through and create header
    for row_header in row:
        for col_header in cols:
            new_col_name = f"{col_header}_{row_header}"
            new_header.append(new_col_name)
    
    return new_header


# Function to drop columns as required.
def drop_columns(df_data, columns_to_drop):
    """
    Drop specified columns from a DataFrame.

    Parameters:
        df_data (pd.DataFrame): The DataFrame from which columns will be dropped.
        columns_to_drop (str or list): A single column header name (str) or a list of column headers names to be dropped.

    Returns:
        pd.DataFrame: DataFrame with remaining columns.
    """
    
    try:
        # Convert columns_to_drop to a list if it's a single string
        if isinstance(columns_to_drop, str):
            columns_to_drop = [columns_to_drop]

        # Drop the columns and return the modified DataFrame
        return df_data.drop(columns=columns_to_drop, axis=1)
    except KeyError as e:
        # Handle the error when one or more columns are not found in the DataFrame
        err = f"Error: One or more columns names not found in the DataFrame: {', '.join(e.args)}"
        log.error(err)
        
    except Exception as e:
        # Handle other unexpected errors
        log.error(f"An unexpected error occurred: {str(e)}")
    
    # Return original dataframe in case of errors
    return df_data

# Function to check data ratio
def is_imbalance(dataframe, class_column='relabeled', min_ratio=0.2, max_ratio=0.8):
    """
    Function to check if the ratio of occurrence of each class data is within the specified range.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame containing class data.
        class_column (str): Name of the column containing class labels.
        min_ratio (float): Minimum acceptable ratio between classes.
        max_ratio (float): Maximum acceptable ratio between classes.
        
    Returns:
        bool: True if the ratio of any class is outside the specified range, False otherwise.
    """
    
    class_counts = dataframe[class_column].value_counts()
    class_ratios = class_counts / class_counts.sum()
    
    for ratio in class_ratios:
        if ratio < min_ratio or ratio > max_ratio:
            return True
    
    return False

# Perform data loading from given csv file
def load_file(file_name:str, folder:str,normalized=True, norm_method='standard', features=[], sampling = True):
    log.info(f"Loading CSV File: {file_name} from {folder}.")
    # Read CSV file
    df_data = read_csv_file(file_name, folder)
    
    # Check for empty dataframe
    if df_data.empty:
        return df_data, df_data
    
    # Check for NaN value
    if df_data[features].isna().any().any() and features:
        log.info(f"At least one NaN is found in {file_name}")
        df_data = df_data.dropna()
    
    # Remove ignored rows
    df_data = filter_ignored_rows(df_data)
    
    #Create X, y for classifier
    X_data = df_data[features] if features else df_data
    y_data = df_data[CSVHeader.RELABELED.value]    
        
    # Normalize data
    if normalized and len(X_data)>0:
        X_data = normalize_data(X_data, norm_method)
        
    # Check if data imbalanced
    if is_imbalance and sampling:
        #Apply sampling algorithm
        X_data, y_data = sampler.sample(X_data, y_data)
    
    # Check for NaN
    if X_data.isna().any().any():
        log.info(f"At least one NaN is found in {file_name}")
        X_data = X_data.dropna()
    
    log.info("---"*5)
    return X_data, y_data


# Function to load all files from give list
def load_all_files(files, ignore_index=True, add_sub_id=False):
    
    # Empty list
    df_list = []
    
    for i in tqdm(range(0,len(files)), desc="Loading CSV Files:"):
        temp = read_csv_file(files[i])
        if add_sub_id:
            temp["sub_id"] = int(files[i].rsplit( "_")[1])
            temp["sub_id"] = temp["sub_id"].astype("uint8")
            #print(i," : ", len(temp))
        df_list.append(temp)
                  
    df = pd.concat(df_list, ignore_index=ignore_index)
    log.info(f"{len(files)} csv files loaded successfully. DataFrame Length:{len(df)}")
    return df
    
# # Perform data loading from given csv file
def load_data(subject:str, csv_files:[str], folder:str,normalized=True, norm_method='standard', features=[], sampling = True):
    log.info(f"Loading CSV Files of subject {subject} from {folder}.")
    
    X = pd.DataFrame()
    y = pd.DataFrame()
    
    for file in csv_files:
        X_temp, y_temp = load_file(file, folder,normalized, norm_method, features, sampling)
        X = pd.concat([X,X_temp])
        y = pd.concat([y, y_temp])
    return X, y


# Perform normalization of given dataframe
def normalize_data(X, method='standard', output_as_df=True):
    if method == 'standard':
        #log.info("Using Standard normalization.")
        scaler = StandardScaler()
    elif method == 'minmax':
        #log.info("Using minmax normalization.")
        scaler = MinMaxScaler()
    else:
        err = "Invalid normalization method. Use 'standard' or 'minmax'."
        log.critical(err)
        raise ValueError(err)
    
    # Transform the DataFrame 
    normalized_data = scaler.fit_transform(X)
    
    if output_as_df:
        # Convert the normalized data back to a DataFrame
        normalized_df = pd.DataFrame(normalized_data, columns=X.columns)
        return normalized_df
    return normalized_data

# Perform count
def count_labels(df):
    # Header name
    header = CSVHeader.RELABELED.value
    # Safety check
    if header not in df.columns:
        return None
    
    # Read the CSV file
    cHW_count = len(df[df[header]==2])
    rHW_count = len(df[df[header]==1])
    other_count = len(df[df[header]==0])

#     labels_counts = {
#         HandWashingType.Compulsive.value:cHW_count,
#         HandWashingType.Routine.value:rHW_count,
#         HandWashingType.NoHandWash.value:other_count
#     }
    
    labels_counts = (other_count,rHW_count,cHW_count)
    
    return labels_counts

# Load x y from dataframe
def load_x_y(df_data,normalized=True, norm_method='standard', features=[]):
    if df_data.empty:
        log.warning("DataFrame is Empty.")
    
    if df_data['acc x'].isnull().values.any():
        print("Dropping NaN")
        df_data = df_data.dropna()
    
    # filter
    df_data = filter_ignored_rows(df_data)
    
    #Create X, y for classifier
    x_data = df_data[features] if features else df_data
    y_data = df_data[CSVHeader.RELABELED.value]    
    
    if normalized and len(x_data)>0:
        x_norm = normalize_data(x_data, norm_method)
        return x_norm, y_data
    
    
# Function to get label distribution
def get_labels_counts(grouped_csv_files,CSV_BLACK_LIST=[]):
    
    #Subjects
    subjects = sorted(grouped_csv_files.keys())
    
    # Empty dict
    labels_counts_by_subject = {}
    
    # Loop and read csv files
    
    print("=="*20)
    print("| Subject | S.NO. | File | Ignored | rHW | cHW | Others | total |")
    print("|---------|-------|------|---------|-----|-----|--------|-------|")
    
    for index, subject in enumerate(subjects):
        
        # Counts
        rHW_count = 0
        cHW_count = 0
        other_count = 0
        
        files_counts = []
    
        files = sorted(grouped_csv_files[subject])
        total = len(files)
        # Loop over files
        for index, file in enumerate(files):
        
            if file in CSV_BLACK_LIST:
                print(f"Skipping CSV file:{file}")
                continue
            
            # Read the CSV file
            df = read_csv_file(file)
                        
            # Filtered ignored types
            df_filtered = filter_ignored_rows(df)
            df_filtered_len = len(df_filtered)
            ignored_total = len(df) - df_filtered_len
            
            # Get labels counts without ignored types
            counts = count_labels(df_filtered)
            
            
            if counts:
                # Get per file counts
                per_file = {file:
                            {"cHW":counts[2],
                            "rHW":counts[1],
                            "NULL":counts[0],
                            "ignored":ignored_total
                            }
                           }

                # Append per files
                files_counts.append(per_file)

                cHW_count += counts[2]
                rHW_count += counts[1]
                other_count += counts[0]
                
                print(f"| {subject} | {index+1} | {file} | {ignored_total} | {counts[1]} | {counts[2]} | {counts[0]} | {df_filtered_len} |")

            else:
                CSV_BLACK_LIST.append(file)
                print(f"File has been black listed {subject} : {file}")
        
                       
        # save counts in dict
        labels_counts = {
                "cHW":cHW_count,
                "rHW":rHW_count,
                "NULL":other_count,
                "files":files_counts
            }
          
        #print(f"For subject {subject}: {labels_counts}")
        
        labels_counts_by_subject[subject] = labels_counts
    
    return labels_counts_by_subject
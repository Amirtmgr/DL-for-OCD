import pandas as pd
import matplotlib.pyplot as plt
from tsfresh.feature_extraction import extract_features
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import numpy as np
#import cupy as cp
import gc

settings = {
    'median': None, 
    'standard_deviation': None, 
    'root_mean_square': None, 
    'maximum': None, 
    'minimum': None,
    'skewness': None, 
    'kurtosis': None,
    'abs_energy': None,
    'sample_entropy': None
    #'sum_values': None, 
    #'mean': None, 
    #'length': None, 
    #'variance': None, 
    #'absolute_maximum': None,   
    #  'last_location_of_maximum': None, 
    #  'first_location_of_maximum': None, 
    #  'last_location_of_minimum': None, 
    #  'first_location_of_minimum': None,
    # ,
    #  "ar_coefficient": [
    #      {"coeff": coeff, "k": k} for coeff in range(4) for k in [10]],
    #  "number_cwt_peaks": [{"n": n} for n in [1, 5]],
    # "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
    # "fft_aggregated": [
    #                 {"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]
    #                 ]
    }

def extract(df, column_id='window_id',default_fc_parameters=settings,n_jobs=12):
    print(f"Extraction Settings = {default_fc_parameters}\nn_jobs = {n_jobs}")
    features = extract_features(df, column_id=column_id, default_fc_parameters=default_fc_parameters,n_jobs=n_jobs)
    return features

    
def compute_fft(dataframe, column, sampling_rate=50):
    """
    Computes the Fast Fourier Transform (FFT) of a specific column in the DataFrame.
    
    Args:
    dataframe (pd.DataFrame): The DataFrame containing sensor data.
    column (str): The column name for which FFT will be computed.
    
    Returns:
    np.array: The frequency values.
    np.array: The corresponding magnitude spectrum.
    """
    signal = dataframe[column].values
    fs = sampling_rate
    
    n = len(signal)
    
    # signal_gpu = cp.asarray(signal)
    # freq_gpu = cp.fft.fftfreq(n, 1/fs)
    # magnitude_gpu = cp.abs(cp.fft.fft(signal_gpu))

    # freqs = cp.asnumpy(freq_gpu)
    # magnitude = cp.asnumpy(magnitude_gpu)

    freqs = np.fft.fftfreq(n, 1/fs)
    magnitude = np.abs(np.fft.fft(signal))
    
    return freqs, magnitude

def plot_fft(freqs, magnitude, column_name):
    """
    Plots the Fast Fourier Transform (FFT) magnitude spectrum.
    
    Args:
    freqs (np.array): The frequency values.
    magnitude (np.array): The magnitude spectrum.
    column_name (str): The name of the column for which FFT was computed.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(freqs, magnitude)
    plt.title(f'FFT of {column_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()



def compute_fft(dataframe, columns):
    """
    Computes the Fast Fourier Transform (FFT) of specified columns in the DataFrame.
    
    Args:
    dataframe (pd.DataFrame): The DataFrame containing sensor data.
    columns (list): List of column names for which FFT will be computed.
    
    Returns:
    dict: A dictionary containing frequency values and corresponding magnitude spectra for each column.
    """
    fs = 50  # Sampling rate
    fft_results = {}
    
    fft_df_list = []
    fft_cols = []

    for column in columns:
        signal = dataframe[column].values
        n = len(signal)
        freqs = np.fft.fftfreq(n, 1/fs)
        magnitude = np.abs(np.fft.fft(signal))
        
        fft_df = pd.DataFrame({
                'frequency': freqs,
                'magnitude': magnitude
                })
        fft_df_list.append(fft_df)
        fft_cols.append(column)
        
    return fft_df_list, fft_cols

def plot_fft_results(fft_results):
    """
    Plots the FFT magnitude spectra for each column.
    
    Args:
    fft_results (dict): A dictionary containing frequency values and magnitude spectra for each column.
    """

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, col in enumerate(fft_results):
        signal = df[col]
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal))
        magnitude = np.abs(fft_vals)

        ax = axes[i]
        ax.plot(fft_freq, magnitude)
        ax.set_title(f'FFT - {col}')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude')

    plt.tight_layout()
    plt.show()


# Function to do feature selection with f_classif
def apply_fclassif(data, labels, k=10):
    """
    Perform feature selection using SelectKBest with f_classif scoring function.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing the features.
        labels (pd.Series): Target variable (labels) corresponding to each sample in 'data'.
        k (int): Number of top k features to select. Default is 10.

    Returns:
        pd.DataFrame: A new DataFrame containing only the selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selected_features = selector.fit_transform(data, labels)
    selected_columns = data.columns[selector.get_support()]
    return pd.DataFrame(selected_features, columns=selected_columns)


def apply_variance_threshold(data, threshold=0.0):
    """
    Perform feature selection using VarianceThreshold.

    Parameters:
        data (pd.DataFrame): Input DataFrame containing the features.
        threshold (float): Features with a variance lower than this threshold will be removed. Default is 0.0.

    Returns:
        pd.DataFrame: A new DataFrame containing only the selected features.
    """
    selector = VarianceThreshold(threshold=threshold)
    selected_features = selector.fit_transform(data)
    selected_columns = data.columns[selector.get_support()]
    return pd.DataFrame(selected_features, columns=selected_columns)

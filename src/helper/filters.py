import pandas as pd
from src.helper.logger import Logger
from scipy.signal import butter, lfilter, filtfilt

    
def low_pass_filter(df, order, fc, cols, fs= 50.0):
    """
    Filter data using low_pass filter.
    
    Args:
    df (pd.DataFarme): Input signal.
    order (int): Order of the filter for smoothness.
    fc (float): Cut-off frequency.
    cols (list): Columns to apply.
    fs (float): Sampling rate in Hz.
    
    Returns:
    Low-pass filtered dataframe.
    """
    
    for col in cols:
        # Numerator (b) and denominator (a) polynomials of the IIR filter. 
        b, a = butter(order, fc, 'lowpass', analog=False, fs=fs)
        # Apply a digital filter forward and backward to a signal.
        df[col] = filtfilt(b, a, df[col])
    return df


def high_pass_filter(df, order, fc, cols, fs=50.0):
    """
    Filter data using high_pass filter.
    
    Args:
    df (pd.DataFarme): Input signal.
    order (int): Order of the filter for smoothness.
    fc (float): Cut-off frequency.
    cols (list): Columns to apply.
    fs (float): Sampling rate in Hz.
    
    Returns:
    High-pass filtered dataframe.
    """
    
    for col in cols:
        # Numerator (b) and denominator (a) polynomials of the IIR filter. 
        b, a = butter(order, fc, 'highpass', analog=False, fs=fs)
        # Apply a digital filter forward and backward to a signal.
        df[col] = filtfilt(b, a, df[col])
    return df

            
def band_pass_filter(df, order, fc_high, fc_low, cols, fs=50.0):
    """
    Filter data using band_pass filter.
    
    Args:
    df (pd.DataFarme): Input signal.
    order (int): Order of the filter for smoothness.
    fc_high (float): Upper cut-off frequency.
    fc_low (float): Lower cut-off frequency.
    cols (list): Columns to apply.
    fs (float): Sampling rate in Hz.
    
    Returns:
    High-pass filtered dataframe.
    """
    Logger.info(f"Signal Filtering with band_pass filter: Order={order} | Upper Threshold={fc_high} | Lower Threshold={fc_low} | Sampling rate = {fs} | Columns = {cols}")
    for col in cols:
        # Numerator (b) and denominator (a) polynomials of the IIR filter. 
        b, a = butter(order, [fc_low, fc_high], 'bandpass', analog=False, fs=fs)
        # Apply a digital filter forward and backward to a signal.
        df[col] = filtfilt(b, a, df[col])
    return df
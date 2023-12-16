import src.helper.df_manager as dfm
import src.helper.data_structures as ds
import src.helper.directory_manager as dm
from src.helper import plotter as pl
from src.utils.config_loader import config_loader as cl
from src.helper.filters import band_pass_filter
from src.helper import sliding_window as sw 
import src.data_processing.features as features
from tsfresh.feature_extraction import MinimalFCParameters
import pandas as pd
import numpy as np
import gc

def clean_all():
    # Get all files
    csv_files = dm.get_files_names()
    grouped_files = ds.group_by_subjects(csv_files)
    
    # Loop through each subjects:
    for sub_id in grouped_files.keys():
        # if sub_id !=3 :
        #     continue

        files = grouped_files[sub_id]
        temp_df = dfm.load_all_files(files, add_sub_id=True)

        # plot signal

        temp_df = clean(temp_df, fft= False) #sub_id==3)

        #plot singal

        # start = 381798-250-1900
        # end = 381798+70

        # dfs = [temp_df.iloc[start:end,1:4].copy(), temp_df.iloc[start:end,4:7].copy()]    
        # pl.plot_subplots(2,1,dfs, ["Accelerometer sensor axes", "Gyroscope sensor axes"] , "Time step", "Sensor Value", "Raw sensors data")

        # Normalize data
        norm_df = dfm.normalize_data(temp_df.iloc[:,1:7], method="minmax")
        #plot signal

        # ndfs = [norm_df.iloc[start:end,1:4].copy(), norm_df.iloc[start:end,4:7].copy()]
        # pl.plot_subplots(2,1,ndfs, ["Accelerometer sensor axes", "Gyroscope sensor axes"] , "Time step", "Sensor Value", "Normalized Sensor Data: Min-Max Scaling")

        # del dfs, ndfs

        norm_df["relabeled"] = temp_df["relabeled"]
        norm_df["datetime"] = temp_df["datetime"]


        # Print 
        print("Windowing data...")
        print(norm_df.info)
        windows, labels, datetimes = sw.get_windows(norm_df, cl.config.dataset.window_size, overlapping_ratio=cl.config.dataset.overlapping_ratio, check_time=True, keep_id=True, keep_date=True)
        print(f"{sub_id}: {len(windows)}")

        
        df_windows = pd.concat(windows, ignore_index=True, axis=0)
        df_labels = pd.DataFrame(labels, columns=["relabeled"])
        df_datetimes = pd.DataFrame(datetimes, columns=["datetime"])
        
        del windows, labels, datetimes, norm_df, temp_df
        gc.collect()

        # Extract Features
        df_features = features.extract(df_windows,settings=MinimalFCParameters(), n_jobs=12)
        df_features["relabeled"] = df_labels["relabeled"]
        df_features["datetime"] = df_datetimes["datetime"]

        del df_windows, df_datetimes, df_labels
        gc.collect()
        
        # save
        file_name = f"OCDetect_{sub_id:02}_features.csv"
        dfm.save_to_csv(df_features, "processed", file_name)
        
    print("Complete cleaning all files.")

def clean(df, fft=False):
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
    

    # Before filtering
    if fft:
        # FFT
        df_fft, cols = ft.compute_fft(df, columns)

        # Plot
        pl.plot_subplots(2,3,df_fft,cols, "Frequency (Hz)", "Magnitude", "FFT Visualization of raw sensors data")
        # Apply Band-pass filter
    

    # Apply Band-pass filter
    df_filtered = band_pass_filter(df, order, fc_high, fc_low, columns, fs)


    # After filtering
    if fft:
        # FFT
        df_filtered_fft, cols = ft.compute_fft(df_filtered, columns)

        # Plot after filtering
        pl.plot_subplots(2,3,df_filtered_fft, cols, "Frequency (Hz)", "Magnitude", "FFT Visualization of filtered sensors data (band-pass-filtered)")

    # Reset index
    df_filtered.reset_index(drop=True, inplace=True)
 
    # Del df
    del df
    gc.collect()

    return df_filtered


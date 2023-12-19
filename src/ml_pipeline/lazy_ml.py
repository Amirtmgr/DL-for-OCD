from datetime import datetime
import gc
import numpy as np
import random
from collections import Counter
import copy
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, TensorDataset
from tabulate import tabulate
import datetime
from src.dl_pipeline import train as t
from src.helper import data_preprocessing as dp
from src.helper import df_manager as dfm
from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
from src.helper.filters import band_pass_filter
from src.helper import object_manager as om
from src.helper import directory_manager as dm
from src.helper.state import State
from src.helper import data_structures as ds
from src.helper import plotter as pl
from src.helper.metrics import Metrics
from src.data_processing import features as ft
from imblearn.under_sampling import OneSidedSelection, NearMiss, RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import classification_report
from src.helper.data_model import TaskType

# Function to load data
def load_features():
    csv_files = dm.get_files_names()
    grouped_files = ds.group_by_subjects(csv_files)
    subjects = list(grouped_files.keys())
    
    # List of dataframes
    df_list = []
    
    # Loop through each subjects:
    for sub_id in subjects:
        # Skip subject 5
        if sub_id == 5:
            continue
        files = grouped_files[sub_id]
        temp_df = dfm.load_all_files(files, add_sub_id=True)
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    
    if cl.config.train.task_type.value == TaskType.rHW_cHW_binary.value:
        print("*********"*20)
        print("rHW vs cHW binary")
        temp_df = df[df["relabeled"] != 0].copy().reset_index(drop=True)
        temp_df['relabeled'].replace(1, 0, inplace=True)
        temp_df['relabeled'].replace(2, 1, inplace=True)
        return temp_df
       
    elif cl.config.train.task_type.value == TaskType.cHW_detection.value:
        print("*********"*20)
        print("Null vs cHW binary")
        df['relabeled'].replace(1, 0, inplace=True)
        df['relabeled'].replace(2, 1, inplace=True)
        return df
    elif cl.config.train.task_type.value == TaskType.HW_detection.value:
        print("*********"*20)
        print("Null vs HW binary")
        df['relabeled'].replace(2, 1, inplace=True)
        return df
    else:
        print("*********"*20)
        print("Null vs cHW vs HW")
        return df
    
def select_features(df):
    selected_df =  ft.apply_variance_threshold(df.drop(columns=["sub_id", "relabeled", "datetime"], axis=1).copy(), threshold=0.9)
    return selected_df

def run():
    

    # Results
    results = {}
    tables = {}
    # Load data
    df = load_features()
    
    selected_df = select_features(df)
    # Split data
    X = selected_df.copy()
    y = df["relabeled"]
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    
    #is_binary = cl.config.dataset.num_classes < 3
    is_binary = cl.config.train.task_type.value < 2
    const = 1 if is_binary else 2
    
    random_seed = cl.config.dataset.random_seed
    n_splits = cl.config.train.cross_validation.k_folds
    shuffle = cl.config.dataset.shuffle
    new_seed = random_seed if shuffle else None
    
    stratified_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=new_seed if shuffle else None)
    print(stratified_kf.get_n_splits(X, y))
    
    if cl.config.train.task_type.value == 3:
        #Selectig 
        ids = [5, 10, 15, 21, 24]
        lazypredict.Supervised.CLASSIFIERS = [lazypredict.Supervised.CLASSIFIERS[i] for i in ids]
        
    Logger.info(f"Training On {len(lazypredict.Supervised.CLASSIFIERS)} Classifiers")
    for i, x in enumerate(lazypredict.Supervised.CLASSIFIERS):
        print(i, x)
        Logger.info(f"{i} {x}")
    
    
    for i, (train_index, val_index) in enumerate(stratified_kf.split(X, y)):
        print("*********"*20)
        #print(train_index)
        #print(val_index)
        
        # Split data
        train_data = X.iloc[train_index]
        train_labels = y.iloc[train_index]
        val_data = X.iloc[val_index]
        val_labels = y.iloc[val_index]

        # start of k-fold
        k_start = datetime.datetime.now()
        Logger.info(f"-------------------")
        Logger.info(f"\nStarting Stratified_k-fold cross-validation on fold no. {i+1} start time: {k_start}...")
        
        # Info
        Logger.info(f"Stratified_k-Fold:{i+1} ===> Train data shape: {train_data.shape} | Train labels shape: {train_labels.shape}")
        Logger.info(f"Stratified_k-Fold:{i+1} ===> Val data shape: {val_data.shape} | Val labels shape: {val_labels.shape}") 
        
        #Selectig initial 10 classifiers


        clf = LazyClassifier(verbose=10, predictions=True, ignore_warnings=True, custom_metric=None)
        models, prediction = clf.fit(train_data, val_data, train_labels, val_labels)
        print(models)
        Logger.info(f"-------------------")
        Logger.info(f"Stratified_k-Fold:{i+1} ===> Duration: {datetime.datetime.now() - k_start}")
        Logger.info(f"-------------------")
        Logger.info(models)
        
        # New table
        table = []
        metrices = []
        cols = prediction.columns.to_list()
        table.append(["ML Algorithm", "Precision", "Sensitivity", "Specificity", "F1-Score", "Accuracy"])
        
        for idx, col in enumerate(cols):
            print(col)
            y_pred = prediction[col]
            #print(prediction[col])
            print("*********"*20)
            print("*********"*20)
            print(classification_report(val_labels, y_pred))
            print("*********"*20)
            print("*********"*20)
            metrics = Metrics(0, is_binary)
            metrics.y_true = val_labels
            metrics.y_pred = y_pred
            metrics.phase = "validation"
            Logger.info(f"[{col}]. Results:")
            print(f"[{col}]. Results:")
            metrics.calculate_metrics()
            metrics.new_save_cm(f"Classifier: {col} | k-Fold: {i+1}")
            #metrics.save_cm(info=f" Classifier: {col} | k-Fold: {i+1}")
            #metrics.save_cm(info=f" Classifier: {col} | k-Fold: {i+1}")
            table.append([col, metrics.precision_score, metrics.recall_score,  metrics.specificity_score, metrics.f1_score, metrics.accuracy])
            metrices.append(metrics)
        
        col = f"Chance [Constant : {const}]"
        dummy_clf = DummyClassifier(strategy='constant', constant=const)
        dummy_clf.fit(train_data, train_labels)
        y_pred = dummy_clf.predict(val_data)
        metrics = Metrics(0, is_binary)
        metrics.y_true = val_labels
        metrics.y_pred = y_pred
        metrics.phase = "validation"
        Logger.info(f"[{col}]. Results:")
        print(f"[{col}]. Results:")
        metrics.calculate_metrics()
        metrics.new_save_cm(f"Classifier: Chance [Constant:{const}] | k-Fold: {i+1}")
        table.append([col, metrics.precision_score, metrics.recall_score,  metrics.specificity_score, metrics.f1_score, metrics.accuracy]) 
        metrices.append(metrics)        
        print("*********"*20)
        print("*********"*20)
        print(f"Results: k-Fold: {i}")
        Logger.info(f"Results: k-Fold: {i}")
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
        Logger.info(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
        tables[i] = table
        results[i] = metrices
        
    Logger.info(f"*********"*20)
    Logger.info(f"End of Stratified_k-fold cross-validation")
    
    # Save results
    msg = om.save_object(results, cl.config.folder, dm.FolderType.results, "results.pkl" )
    Logger.info(msg)
    msg = om.save_object(tables, cl.config.folder, dm.FolderType.results, "tables.pkl" )
    Logger.info(msg)
    Logger.info("******"*20)
    Logger.info(f"Results:")
    print(f"Results:")
    for i, table in tables.items():
        print(f"Results: k-Fold: {i+1}")
        Logger.info(f"Results: k-Fold: {i+1}")
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
        Logger.info(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
    
    
'''
def run(personalized_subject = 3):

    print("======"*20)
    Logger.info("======"*20)
    # start
    start =  datetime.now()
    Logger.info(f"Dummy Classifier Personalization on Subject: {personalized_subject} Start time: {start}")
    print(f"Dummy Classifier Personalization on Subject: {personalized_subject} Start time: {start}")


    is_binary = cl.config.dataset.num_classes < 3
    shelf_name = cl.config.dataset.name
    random_seed = cl.config.dataset.random_seed
    shuffle = cl.config.dataset.shuffle
    train_ratio = cl.config.dataset.train_ratio
    n_jobs = -1

    # Load python dataset
    X_train, X_infer, y_train, y_infer, scaler = prepare_data(shelf_name, personalized_subject, train_ratio, random_seed, shuffle)

    # Create dummy classifier
    stratgeies = ['stratified', 'most_frequent', 'prior', 'uniform', 'constant (0)', 'constant (1)']

    # Results
    results = {}
    table = []
    table.append(["Strategy", "Precision", "Sensitivity", "Specificity", "F1-Score", "Accuracy"])

    # Different strategies
    for strategy in stratgeies:
        Logger.info(f"Dummy Classifier Personalization ===> Strategy: {strategy}")
        metrics = classify(X_train, X_infer, y_train, y_infer, strategy, is_binary)
        results[strategy] = metrics
        table.append([strategy, metrics.precision_score, metrics.recall_score,  metrics.specificity_score, metrics.f1_score, metrics.accuracy])

        # End
    end = datetime.now()
    Logger.info(f"Dummy Classifier Personalization on subject: {personalized_subject} ===> Duration: {end - start}")
    print(f"Dummy Classifier Personalization on subject: {personalized_subject} ===> Duration: {end - start}")

    msg = om.save_object(results, cl.config.folder, dm.FolderType.results, "results.pkl" )
    Logger.info(msg)
    Logger.info("******"*20)
    print("******"*20)
    Logger.info(f"Dummy Classifier Personalization on subject [{personalized_subject}] ===> Results:")
    print(f"Dummy Classifier Personalization on subject [{personalized_subject}] ===> Results:")
    Logger.info(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))



# Function to run dummy classifier
def classify(X_train, X_infer, y_train, y_infer, strategy, is_binary):
    # Create dummy classifier
    if strategy == 'constant (0)':
        dummy_clf = DummyClassifier(strategy='constant', constant=0)
    elif strategy == 'constant (1)':
        dummy_clf = DummyClassifier(strategy='constant', constant=1)
    else:
        dummy_clf = DummyClassifier(strategy=strategy)
    
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_infer)

    metrics = Metrics(0, is_binary)
    metrics.y_true = y_infer
    metrics.y_pred = y_pred
    metrics.phase = "inference"
    Logger.info(f"[{strategy}]. Results:")
    print(f"[{strategy}]. Results:")
    metrics.calculate_metrics()
    return metrics

'''
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
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, LeaveOneGroupOut
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

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
    
    if cl.config.train.task_type.value == -1:
        print("*********"*20)
        print("rHW vs cHW binary")
        temp_df = df[df["relabeled"] != 0].copy().reset_index(drop=True)
        temp_df['relabeled'].replace(1, 0, inplace=True)
        temp_df['relabeled'].replace(2, 1, inplace=True)
        return temp_df
       
    elif cl.config.train.task_type.value == 0:
        print("*********"*20)
        print("Null vs cHW binary")
        df['relabeled'].replace(1, 0, inplace=True)
        return df
    elif cl.config.train.task_type.value == 1:
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


def get_model(name, class_weight, const, verbose=0, **kwargs):
    if name == 'logistic_regression':
        model = LogisticRegression(class_weight=class_weight,max_iter=2000, verbose=verbose, n_jobs=-1, **kwargs)
    elif name == 'random_forest':
        model = RandomForestClassifier(class_weight=class_weight,max_features="log2", max_depth=10, verbose=verbose, n_jobs=-1,**kwargs)
    elif name == 'gradient_boosting':
        model = GradientBoostingClassifier(verbose=verbose,**kwargs)
    elif name == 'svm':
        model = SVC(class_weight=class_weight,verbose=verbose, **kwargs)
    elif name == 'kmeans':
        model = KMeans(n_clusters=const+1,verbose=verbose, **kwargs)
    elif name == 'neuralnetwork':
        model = MLPClassifier(verbose=verbose,**kwargs)
    elif name == 'SGDClassifier':
        model = SGDClassifier(class_weight=class_weight,penalty='elasticnet', l1_ratio=0.5,verbose=verbose, n_jobs=-1, early_stopping=True, **kwargs)
    elif name == 'MultinomialNB':
        model = MultinomialNB()
    elif name == 'PassiveAggressiveClassifier':
        model = PassiveAggressiveClassifier(class_weight=class_weight,verbose=verbose, n_jobs=-1)
    elif name ==  'Perceptron':
        model = Perceptron(class_weight=class_weight,verbose=verbose, n_jobs=-1)
    elif name == 'Dummy':
        model = DummyClassifier(strategy='constant', constant=const)
        
    else:
        error = "Invalid classifier choice.\
        Supported classifiers: logistic_regression, random_forest, \
        gradient_boosting, svm, kmeans, neuralnetwork, MultinomialNB, PassiveAggressiveClassifier."
        Logger.critical(error)
        raise ValueError(error)
    
    Logger.info(f"ML model {name} initialized.")
    return model


def run():
    # List models
    all_models = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm', 'kmeans', 'neuralnetwork', 'SGDClassifier', 'PassiveAggressiveClassifier', 'Perceptron', 'Dummy']
    indices = [0,1,3,5,6,8,9]
    models = all_models #[all_models[i] for i in indices]
    for i, model in enumerate(models):
        print(f"{i}.{model}")
        
    # Results
    results = {}
    tables = {}
    all_trained_models = {}
    
    # Load data
    df = load_features()
    
    selected_df = select_features(df)
    selected_df["relabeled"] = df["relabeled"]
    selected_df["sub_id"] = df["sub_id"]
    
    # Personalized data
    personalized_subs = cl.config.dataset.personalized_subjects
    personalized_df = selected_df[selected_df['sub_id'].isin(personalized_subs)].reset_index(drop=True)

    # Train data
    trained_df = selected_df[~selected_df['sub_id'].isin(personalized_subs)].reset_index(drop=True)

    del df, selected_df 
    gc.collect()
     
    # Training data
    X = trained_df.drop(columns=["sub_id", "relabeled"], axis=1).copy()
    y = trained_df["relabeled"].copy()
    subs = trained_df["sub_id"].copy()
    
    # # Personalized data
    X_p = personalized_df.drop(columns=["sub_id", "relabeled"], axis=1).copy()
    y_p = personalized_df["relabeled"].copy()
    subs_p = personalized_df["sub_id"].copy()
    
    del personalized_df
    gc.collect()
    
    #is_binary = cl.config.dataset.num_classes < 3
    is_binary = cl.config.train.task_type.value < 2
    const = 1 if is_binary else 2
    
    random_seed = cl.config.dataset.random_seed
    n_splits = cl.config.train.cross_validation.k_folds
    shuffle = cl.config.dataset.shuffle
    new_seed = random_seed if shuffle else None
    
    # Cross validation
    cv_name = cl.config.train.cross_validation.name
    
    if cv_name== "losocv":
        cv = LeaveOneGroupOut()
        folds = cv.split(X, y, subs)
    elif cv_name== "kfold":
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=new_seed if shuffle else None)
        folds = cv.split(X, y)
    elif cv_name== "stratified":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=new_seed if shuffle else None)
        folds = cv.split(X, y)
    else:
        error = "Invalid cross validation choice.\
        Supported cross validations: losocv, kfold, stratified."
        Logger.critical(error)
        raise ValueError(error)
    
    # Start
    for kfold, (train_index, val_index) in enumerate(folds):
        Logger.info(f"-------------------")
        Logger.info(f"Stratified_k-Fold:{kfold+1} ===> Start")
        # Split data
        train_data = X.iloc[train_index]
        train_labels = y.iloc[train_index]
        val_data = X.iloc[val_index]
        val_labels = y.iloc[val_index]

        # start of k-fold
        k_start = datetime.datetime.now()
        Logger.info(f"-------------------")
        Logger.info(f"\nStarting Stratified_k-fold cross-validation on fold no. {kfold+1} start time: {k_start}...")
        
        # Info
        Logger.info(f"Stratified_k-Fold:{kfold+1} ===> Train data shape: {train_data.shape} | Train labels shape: {train_labels.shape}")
        Logger.info(f"Stratified_k-Fold:{kfold+1} ===> Val data shape: {val_data.shape} | Val labels shape: {val_labels.shape}") 
        
        #Selectig initial 10 classifiers

        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weight_dict = {num: weight for num, weight in enumerate(class_weights)}
        print(f"Class weights: {class_weight_dict}")
        Logger.info(f"Class weights: {class_weight_dict}")
        
        # New table
        val_table = []
        val_metrices = []
        inference_table = []
        inference_metrices = []
        trained_models = []
        
        val_table.append(["Classifier", "Precision", "Sensitivity", "Specificity", "F1-Score", "Accuracy"])
        inference_table.append(["Classifier", "Precision", "Sensitivity", "Specificity", "F1-Score", "Accuracy"])
        
        # Loop over classifiers
        for model in models:
            Logger.info(f"Stratified_k-Fold:{kfold+1} ===> Model: {model}")
            # Get model
            clf = get_model(model, class_weight_dict, const)
            # Train model
            clf.fit(train_data, train_labels)
            # Predict
            y_pred = clf.predict(val_data)
            # Metrics
            val_metrics = Metrics(0, is_binary)
            val_metrics.y_true = val_labels
            val_metrics.y_pred = y_pred
            val_metrics.phase = "validation"
            Logger.info(f"[{model}]. Results:")
            print(f"[{model}]. Results:")
            val_metrics.calculate_metrics()
            val_metrics.new_save_cm(f"Classifier: {model} | k-Fold: {kfold+1}")
            #metrics.save_cm(info=f" Classifier: {model} | k-Fold: {i+1}")
            #metrics.save_cm(info=f" Classifier: {model} | k-Fold: {i+1}")
            val_table.append([model, val_metrics.precision_score, val_metrics.recall_score,  val_metrics.specificity_score, val_metrics.f1_score, val_metrics.accuracy])
            val_metrices.append(val_metrics)
            # Save model
            trained_models.append(clf)
            
            # Inference
            infer_y_pred = clf.predict(X_p)
            infer_metrics = Metrics(0, is_binary)
            infer_metrics.y_true = y_p
            infer_metrics.y_pred = infer_y_pred
            infer_metrics.phase = "inference"
            Logger.info(f"[{model}]. Inference  Results:")
            print(f"[{model}]. Results:")
            infer_metrics.calculate_metrics()
            inference_metrices.append(infer_metrics)
            inference_table.append([model, infer_metrics.precision_score, infer_metrics.recall_score,  infer_metrics.specificity_score, infer_metrics.f1_score, infer_metrics.accuracy])
            
            
       
        # End of k-fold
        Logger.info(f"-------------------")
        Logger.info(f"Stratified_k-Fold:{kfold+1} ===> Duration: {datetime.datetime.now() - k_start}")
        Logger.info(f"-------------------")
        
        print(f"Results: k-Fold: {kfold}")
        Logger.info(f"Results: k-Fold: {kfold}")
        print(tabulate(val_table, headers="firstrow", tablefmt="fancy_grid"))
        print(tabulate(inference_table, headers="firstrow", tablefmt="fancy_grid"))
        
        Logger.info(f"Validataion Results: k-Fold: {kfold}")
        Logger.info(tabulate(val_table, headers="firstrow", tablefmt="fancy_grid"))
        Logger.info(f"Inference results: k-fold: {kfold}")
        Logger.info(tabulate(inference_table, headers="firstrow", tablefmt="fancy_grid"))
        tables[kfold] = {}
        tables[kfold]["val"] = val_table
        tables[kfold]["infer"] = inference_table
        
        results[kfold] = {}
        results[kfold]["validation"] = val_metrices
        results[kfold]["inference"] = inference_metrices
        all_trained_models[kfold] = trained_models
        
    Logger.info(f"*********"*20)
    Logger.info(f"End of Stratified_k-fold cross-validation")
    
    # Save results
    msg = om.save_object(results, cl.config.folder, dm.FolderType.results, "results.pkl" )
    Logger.info(msg)
    msg = om.save_object(tables, cl.config.folder, dm.FolderType.results, "tables.pkl" )
    Logger.info(msg)
    msg = om.save_object(all_trained_models, cl.config.folder, dm.FolderType.results, "trained_models.pkl" )
    Logger.info(msg)
    Logger.info("******\n"*5)
    print("******\n"*5)
    Logger.info(f"Final Results:")
    print(f"Final Results:")
    for kfold, tabs in tables.items():
        for phase, tab in tabs.items():
            print(f"Scores: k-Fold: {kfold+1} | Phase: {phase}")
            Logger.info(f"Scores: k-Fold: {kfold+1} | Phase: {phase}")
            print(tabulate(tab, headers="firstrow", tablefmt="fancy_grid"))
            Logger.info(tabulate(tab, headers="firstrow", tablefmt="fancy_grid"))
    
    # Save models
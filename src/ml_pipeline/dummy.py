from datetime import datetime
import gc
import numpy as np
import random
from collections import Counter
import copy

import torch
from torch.utils.data import ConcatDataset, TensorDataset
from tabulate import tabulate

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
from imblearn.under_sampling import OneSidedSelection, NearMiss, RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier

def prepare_data(shelf_name, personalized_subject, inference_ratio, random_seed, shuffle):
    X_dict, y_dict = dp.load_shelves(shelf_name, personalized_subject)

    X_personalized = X_dict[personalized_subject]
    y_personalized = y_dict[personalized_subject]

    Logger.info(f"Resampled with random_state: {random_seed}")
    X_personalized, y_personalized = resample(X_personalized, y_personalized, random_state=random_seed)
    Logger.info(f"Total samples : | X_personalized shape: {X_personalized.shape} | y_personalized shape: {y_personalized.shape}")

    del X_dict, y_dict
    gc.collect()

    # Split data
    if shuffle:
        X_infer, X_train, y_infer, y_train = train_test_split(X_personalized, y_personalized, train_size= inference_ratio,stratify=y_personalized, shuffle=True, random_state=random_seed)
    else:
        X_infer, X_train, y_infer, y_train = train_test_split(X_personalized, y_personalized, train_size= inference_ratio, shuffle=False)
       
    Logger.info(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    Logger.info(f"X_infer shape: {X_infer.shape} | y_infer shape: {y_infer.shape}")
    # Shape
    samples, time_steps, channels = X_train.shape
    Logger.info(f"Samples: {samples} | Time steps: {time_steps} | Channels: {channels}")

    # Convert 3D to 2D
    Logger.info(f"Dummy Classifier Personalization ===> Reshaping inputs...")
    X_train = X_train.reshape(-1, X_train.shape[2])
    X_infer = X_infer.reshape(-1, X_infer.shape[2])
    y_train = np.repeat(y_train, time_steps )
    y_infer = np.repeat(y_infer, time_steps )

    Logger.info(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    Logger.info(f"X_infer shape: {X_infer.shape} | y_infer shape: {y_infer.shape}")

    # Scale data
    if cl.config.dataset.scaler_type:
        Logger.info(f"Dummy Classifier Personalization ===> Scaling dataframes...")
        scaler = dp.get_scaler()
        X_train = scaler.fit_transform(X_train)
        # Fit transform with new scaler
        X_infer = scaler.transform(X_infer)
    else:
        scaler = None

    return X_train, X_infer, y_train, y_infer, scaler

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


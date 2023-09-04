import pandas as pd
import os
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN
from src.utils.config_loader import config_loader as cl 
from src.helper.logger import Logger as log
from src.helper.data_model import CSVHeader, HandWashingType

# Function that combines over and undersampling
def sample(X,y, sampling_factor=1.5, random_state = 55,n_jobs=-1 ):
    """
    sampling_factor(float): Data grows by this factor times the second minority class.
    
    returns: Oversampled X_sample, y_sample
    """
    counts = y.value_counts().values.tolist()
    
    # Perfom sampling only for more than one label.
    if len(counts) < 2:
        return underSample(X,y, sampling_factor, random_state)
    
    if counts[1]<6:
        return X,y
    
    if len(counts) == 3:
        if counts[2] < 6:
            return X,y
    
    log.info(f"Original distribution: {Counter(y)}")
    X_over, y_over = overSample(X,y, sampling_factor, random_state,n_jobs)
    X_sample, y_sample = underSample(X_over,y_over, sampling_factor, random_state,n_jobs)
    
    return X_sample, y_sample


# Function to oversample minority data to
def overSample(X,y, sampling_factor=1.5, random_state = 55, bias =2, n_jobs=-1 ):
    """
    sampling_factor(float): Data grows by this factor times the second minority class.
    bias(int): Case when 0 and 2 or 1 labels only present.
    returns: Oversampled X_over, y_over
    """
    labels = y.value_counts().keys().tolist()
    counts = y.value_counts().values.tolist()
    
    if len(counts) == 3:
        mid_count = counts[1]
        over_sampling_strategy = {
            1: int(mid_count*sampling_factor),
            2: int(mid_count*sampling_factor)
        }

    elif len(counts) == 2:
        over_sampling_strategy = {
            labels[1]: int(counts[1]*(sampling_factor+bias))
        }
    
    # Apply SMOTE
    oversampler = SMOTE(sampling_strategy=over_sampling_strategy, random_state = random_state,n_jobs=n_jobs)
    X_over, y_over = oversampler.fit_resample(X, y)
    
    log.info(f"Class distribution after oversampling: {Counter(y_over)}")
    
    return X_over, y_over


# Function to under sample majority data
def underSample(X,y,sampling_factor=1.5, random_state = 55, n_jobs=-1):
    """
    sampling_factor(float): Data decrease by this factor times the second minority class.
    
    returns: Sampled X_nm, y_nm
    """
    counts = y.value_counts().values
    max_count = max(counts)
    
    under_sampling_strategy = {
        0: int(counts[0]/sampling_factor)
    }
    
#     # Apply RandomUnderSampler
#     random_sampler = RandomUnderSampler(sampling_strategy=under_sampling_strategy,random_state=random_state)
#     X_rus, y_rus = random_sampler.fit_resample(X, y)
    
#     log.info(f"Class distribution after RandomUnderSampler: {Counter(y_rus)}")
    X_rus, y_rus = X, y
    # Apply NearMissSampler
    nearmiss = NearMiss(version=3,sampling_strategy='majority',n_jobs=n_jobs)
    X_nm, y_nm = nearmiss.fit_resample(X_rus, y_rus)
    log.info(f"Class distribution after NearMiss UnderSampling: {Counter(y_nm)}")
    return X_nm, y_nm
    
    
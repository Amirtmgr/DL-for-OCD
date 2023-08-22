import pandas as pd
import os
from logger import logger
from collections import Counter
from data_model import Metrics, CSVHeader,HandWashingType
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN

# Function that combines over and undersampling
def sample(X,y, sampling_factor=1.25, random_state = 55 ):
    """
    sampling_factor(float): Data grows by this factor times the second minority class.
    
    returns: Oversampled X_sample, y_sample
    """
    
    # Perfom sampling only for more than one label.
    if len(y.value_counts().values.tolist()) < 2:
        return X, y
    
    logger.info(f"Original distribution: {Counter(y)}")
    X_over, y_over = overSample(X,y, sampling_factor, random_state)
    X_sample, y_sample = underSample(X_over,y_over, sampling_factor, random_state)
    
    return X_sample, y_sample


# Function to oversample minority data to
def overSample(X,y, sampling_factor=1.5, random_state = 55, bias =2 ):
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
    oversampler = SMOTE(sampling_strategy=over_sampling_strategy, random_state = random_state)
    X_over, y_over = oversampler.fit_resample(X, y)
    
    logger.info(f"Class distribution after oversampling: {Counter(y_over)}")
    
    return X_over, y_over


# Function to under sample majority data
def underSample(X,y,sampling_factor=1.5, random_state = 55 ):
    """
    sampling_factor(float): Data decrease by this factor times the second minority class.
    
    returns: Sampled X_nm, y_nm
    """
    max_count = max(y.value_counts().values)
    under_sampling_strategy = {
        0: int(max_count/3)
    }
    
    # Apply RandomUnderSampler
    random_sampler = RandomUnderSampler(sampling_strategy=under_sampling_strategy,random_state=random_state)
    X_rus, y_rus = random_sampler.fit_resample(X, y)
    
    logger.info(f"Class distribution after RandomUnderSampler: {Counter(y_rus)}")
    
    # Apply NearMissSampler
    nearmiss = NearMiss(version=3,sampling_strategy='majority')
    X_nm, y_nm = nearmiss.fit_resample(X_rus, y_rus)
    logger.info(f"Class distribution after NearMiss UnderSampling: {Counter(y_nm)}")
    return X_nm, y_nm
    
    
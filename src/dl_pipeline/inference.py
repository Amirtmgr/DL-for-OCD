import datetime
import gc
import numpy as np
import random
from collections import Counter

import torch
from torch.utils.data import ConcatDataset, TensorDataset

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

from imblearn.under_sampling import OneSidedSelection, NearMiss, RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, train_test_split

def infer(device, multi_gpu=False):
    print("Device:", device)

    print("======"*5)
    # start
    start = datetime.datetime.now()
    Logger.info(f"Inference Start time: {start}")
    print(f"Inference Start time: {start}")

    is_binary = cl.config.dataset.num_classes < 3
    shelf_name = cl.config.dataset.name
    random_seed = cl.config.dataset.random_seed
    shuffle = cl.config.dataset.shuffle
    new_seed = random_seed if shuffle else None
    cv_name = cl.config.dataset.cross_validation.name
    personalized_subject = str(cl.config.dataset.personalized_subject)

    # Load python dataset
    X_dict, y_dict = dp.load_shelves(shelf_name)

    X_infer = X_dict[personalized_subject]
    y_infer = y_dict[personalized_subject]

    Logger.info(f"X_infer shape: {X_infer.shape} | y_infer shape: {y_infer.shape}")

    # TO do
    #         


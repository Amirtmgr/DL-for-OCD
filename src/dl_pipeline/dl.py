import numpy as np
import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from tabulate import tabulate

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler, MinMaxScaler, RobustScaler

from src.helper import directory_manager as dm
from src.helper import df_manager as dfm
from src.helper import data_structures as ds
from src.helper.data_model import CSVHeader, HandWashingType, LABELS
from src.helper.sliding_window import get_windows, process_dataframes
from src.helper.data_model import TaskType

from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
from src.dl_pipeline.architectures.CNN import CNNModel
from src.dl_pipeline import train as t
from src.helper import data_preprocessing as dp
from src.dl_pipeline import validate as v
from src.dl_pipeline import personalize as p


# Function to setup random seed
def setup_random_seed():
    seed = cl.config.train.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Function to setup CUDA
def setup_cuda():
    # Check CUDA
    if not torch.cuda.is_available():
        Logger.error("CUDA is not available. Using CPU only.")
        return "cpu"

    if cl.config.train.device == "cuda":
        Logger.info("Using CUDA device.")
        num_gpus = torch.cuda.device_count()
        Logger.info(f"Number of GPUs: {num_gpus}")
        
        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        cudnn.fastest = True
        cudnn.deterministic = True
        return device

# Function to train model and return state

def train():
    # Setup CUDA
    device = setup_cuda()
    Logger.info(f"Device: {device}")

    # Setup random seed
    setup_random_seed()

    # Check if Multi-GPUs
    #multi_gpu = t.ddp_setup()
        
    multi_gpu = cl.config.world_size > 1

    Logger.info(f"Using {torch.cuda.device_count()} GPUs!")

    cv = cl.config.train.cross_validation.name
    task_type = TaskType(cl.config.dataset.task_type)
    cl.config.train.task_type = task_type
    num_classes = 2 if task_type.value !=TaskType.Multiclass_classification.value else 3
    cl.config.dataset.num_classes = num_classes
    cl.config.architecture.num_classes = num_classes

    if cl.config.dataset.personalization:
        if cl.config.train.ensemble:
            p.ensemble(device, multi_gpu)
        else:
            personalize_all(device, multi_gpu)
    else: 
        if cv == "loso"  or cv == "kfold":
            v.subwise_k_fold_cv(device, multi_gpu)
        elif cv == "stratified":
            v.stratified_k_fold_cv(device, multi_gpu)
        elif cv == "personalized":
            personalize_all()
        else:
            Logger.error("Cross-validation method not supported.")
            raise ValueError("Cross-validation method not supported.")
        
    # Clean up
    #t.ddp_destroy()



def personalize_all(device, multi_gpu):
    before_table = []
    table_based_on_loss = []
    table_based_on_f1 = []

    header = ["Subject", "F1-Score", "Precision", "Recall", "Specificity", "Accuracy"]
    before_table.append(header)
    table_based_on_f1.append(header)
    table_based_on_loss.append(header)

    for i in cl.config.dataset.personalized_subjects:
        cl.config.dataset.personalized_subject = i
        Logger.info(f"****************************")
        Logger.info(f"Personalizing for subject {i}")
        infer_metrics_0, infer_metrics_1, infer_metrics_2 = p.run(device, multi_gpu)
        before_table.append([i, f"{infer_metrics_0.f1_score:.2f}", f"{infer_metrics_0.precision_score:.2f}", f"{infer_metrics_0.recall_score:.2f}", f"{infer_metrics_0.specificity_score:.2f}", f"{infer_metrics_0.accuracy:.2f}"])
        table_based_on_f1.append([i, f"{infer_metrics_1.f1_score:.2f}", f"{infer_metrics_1.precision_score:.2f}", f"{infer_metrics_1.recall_score:.2f}", f"{infer_metrics_1.specificity_score:.2f}", f"{infer_metrics_1.accuracy:.2f}"])
        table_based_on_loss.append([i, f"{infer_metrics_2.f1_score:.2f}", f"{infer_metrics_2.precision_score:.2f}", f"{infer_metrics_2.recall_score:.2f}", f"{infer_metrics_2.specificity_score:.2f}", f"{infer_metrics_2.accuracy:.2f}"])
        
    Logger.info(f"****************************")
    Logger.info("Results: Before Personalization")
    Logger.info(tabulate(before_table, headers="firstrow", tablefmt="fancy_grid"))
    Logger.info(f"****************************")
    Logger.info("Results: Based on F1-Score")
    Logger.info(tabulate(table_based_on_f1, headers="firstrow", tablefmt="fancy_grid"))
    Logger.info(f"****************************")
    Logger.info("Results: Based on Val Loss")
    Logger.info(tabulate(table_based_on_loss, headers="firstrow", tablefmt="fancy_grid"))
    Logger.info(f"****************************")
    Logger.info(f"Congratulations! You have completed the personalization process.")

    results = before_table + table_based_on_f1 + table_based_on_loss
    df = pd.DataFrame(results[1:], columns=results[0])
    result_path = os.path.join(cl.config.results_path, f"{cl.config.folder}_tasktype_{cl.config.train.task_type}.csv")
    df.to_csv(result_path, index=False)
    Logger.info(f"Results saved at{result_path}")
    
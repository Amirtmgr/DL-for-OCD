# Description: Main file to run the project.
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

# Ignore warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import the necessary packages
import os
import argparse
from pathlib import Path
from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
import src.dl_pipeline.dl as dl
import src.helper.plotter as plot
from src.data_processing import cleaner as cln
from src.data_processing import process as prc
from collections import Counter
from src.helper import visual as vis
from src.helper.data_model import TaskType
from src.ml_pipeline import dummy as dmy
from src.ml_pipeline import ml, lazy_ml

def main():
    # Parse the arguments
    arg_parser = argparse.ArgumentParser(description="Path for the config file.")
    
    # Path for yaml config file
    arg_parser.add_argument(
        'config',
        default='None',
        help='The Configuration file in yaml file format.')
    
    arg_parser.add_argument(
        'method',
        default='None',
        help='Method to apply \'ml\' for Machine Learning or \'dl\' for DeepLearning.')

    args = arg_parser.parse_args()

    # Set current working directory
    cl.set_main_path(os.getcwd())
    # Load the config file
    cl.load_config(args.config)

    if cl.config:
        print('Config file loaded successfully.')
    else:
        print('Config file is empty.')
    
    print("Device:", cl.config.train.device)
    print("Models path:", cl.config.models_path)
    #cl.export_to_environment(["main_path"])
    #print(os.environ['main_path'])

    # Create Logger
    Logger.info("Logger created.")

    # Log config
    log_config(cl.config_dict)
    # Task Type
    task_type = TaskType(cl.config.dataset.task_type)
    cl.config.train.task_type = task_type
    Logger.info(f"Task type: {task_type}")
    
    # Perform data cleaning and preprocessing
    #cln.clean_all()

    # Prepare datset
    # prc.prepare_datasets("OCDetect_raw_1950")
    # X, y  = prc.load_dataset(15, "OCDetect_raw_1950")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))
    
    # cl.config.dataset.name = "OCDetect_Export"
    # cl.config.dataset.window_size = 380
    
    # # Prepare dataset
    # prc.make_datasets("OCDetect_sep_380")
    # X, y  = prc.load_dataset(30, "OCDetect_sep_380")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))

    # print("--------------------------------------------------")

    # cl.config.dataset.window_size = 285
    # prc.make_datasets("OCDetect_sep_285")
    # X, y  = prc.load_dataset(30, "OCDetect_sep_285")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))

    # print("--------------------------------------------------")

    # cl.config.dataset.window_size = 190
    # prc.make_datasets("OCDetect_sep_190")
    # X, y  = prc.load_dataset(30, "OCDetect_sep_190")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))
    
    # print("--------------------------------------------------")

    # cl.config.dataset.window_size = 95
    # prc.make_datasets("OCDetect_sep_95")
    # X, y  = prc.load_dataset(30, "OCDetect_sep_95")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))

    # print("--------------------------------------------------")
    # cl.config.dataset.window_size = 475
    # prc.make_datasets("OCDetect_sep_475")
    # X, y  = prc.load_dataset(30, "OCDetect_sep_475")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))

    # cl.config.dataset.window_size = 950
    # prc.make_datasets("OCDetect_sep_950")
    # X, y  = prc.load_dataset(30, "OCDetect_sep_950")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))

    # cl.config.dataset.window_size = 1425
    # prc.make_datasets("OCDetect_sep_1425")
    # X, y  = prc.load_dataset(30, "OCDetect_sep_1425")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))

    # cl.config.dataset.window_size = 1140
    # prc.make_datasets("OCDetect_sep_1140")
    # X, y  = prc.load_dataset(30, "OCDetect_sep_1140")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))

    # cl.config.dataset.window_size = 1900
    # prc.prepare_datasets("OCDetect_raw_380")
    # X, y  = prc.load_dataset(30, "OCDetect_raw_380")
    # print(X.shape)
    # print(y.shape)
    # print(Counter(y))

    # Prepare sub-datasets
    #prc.prepare_subset()
    
    
    # Perform DL pipeline
    if cl.config.world_size == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    elif cl.config.world_size == 4:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        
    #dl.train()
    task_type = TaskType(cl.config.dataset.task_type)
    cl.config.train.task_type = task_type
    num_classes = 2 if task_type.value !=TaskType.Multiclass_classification.value else 3 
    cl.config.dataset.num_classes = num_classes
    cl.config.architecture.num_classes = num_classes

    if task_type == TaskType.rHW_cHW_binary:
        msg = "==============Binary classification============="
        msg += "\n=============== rHW vs cHW =============="
        cl.config.dataset.labels = ["rHW", "cHW"]
    elif task_type == TaskType.cHW_detection:
        msg = "==============Binary classification============="
        msg += "\n=============== Null vs cHW =============="
        cl.config.dataset.labels = ["Null", "cHW"]
    elif task_type == TaskType.HW_detection:
        msg = "==============Binary classification============="
        msg += "\n=============== Null vs HW =============="
        cl.config.dataset.labels = ["Null", "HW"]
    elif task_type == TaskType.HW_classification:
        msg = "==============Multi-class classification============="
        msg += "\n=============== rHW vs cHW =============="
        cl.config.dataset.labels = ["rHW", "cHW"]
    elif task_type == TaskType.Multiclass_classification:
        msg = "==============Multi-class classification========="
        cl.config.dataset.labels = ["Null", "rHW", "cHW"]
    else: 
        raise ValueError("Task type not found in config file.")
    
    Logger.info(msg)
    print(msg)
    cl.print_config_dict()

    # Datasets selection
    if args.method == 'dl':
        cl.config.dataset.name = "OCDetect_sep_380"
        dl.train()
    elif args.method == 'ml':
        cl.config.dataset.name = "processed"
        cl.config.train.cross_validation.name = "losocv"
        ml.run()
    elif args.method == 'lazy':
        cl.config.dataset.name = "processed"
        lazy.run()
    else:
        raise ValueError("Method not found. Use \'dl\' or \'ml\'")
    
    
    Logger.info("Training Completed!")   
    # Run dummy classifier
    # for sub in [3, 15, 18, 30]:
    #     dmy.run(str(sub))
    # Visualize all plots
    #vis.show()


def log_config(config_dict):
    """Print config dictionary
    """
    Logger.info("---------------------" * 5)
    Logger.info("Config Dictionary:")
    for k, v in config_dict.items():
        if isinstance(v, dict):
            Logger.info(f"{k}:")
            for k1, v1 in v.items():
                Logger.info(f"\t{k1}: {v1}")
        else:
            Logger.info(f"{k}: {v}")

    print("---------------------" * 5)

if __name__ == '__main__':
    main()

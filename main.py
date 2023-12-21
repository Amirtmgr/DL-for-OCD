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
        help='The Configuration file in yaml file format must be provided. Select file from the `config` folder.')
    
    arg_parser.add_argument('--t',
        dest= 'task_type',
        type=int,
        choices=[1,2,3,4,5],
        metavar='task_type',
        default=None,
        help='Optional. To override `task_type` in config file provide Task type to perform. Task-1: Null vs HW, Task-2:rHW vs cHW, Task-3:Null vs cHW, Task-4: Null vs rHW vs cHW, Task-5: DL personalization')
    
    arg_parser.add_argument('--m',
        dest= 'method',
        type=str,
        choices=['ml', 'dl'],
        metavar='method',
        default='None',
        help='Method to apply \'ml\' for Machine Learning or \'dl\' for DeepLearning.')

    arg_parser.add_argument('--a',
        dest= 'architecture',
        type=str,
        choices=['cnn_transformer', 'deepconvlstm', 'tinyhar'],
        metavar='architecture',
        default=None,
        help='Architecture to use for DeepLearning. Make sure to include pre-processed data inside the `data` folder or use `--d` to perform data preprocessing only.')    
    
    arg_parser.add_argument('--d',
        dest= 'data_preprocessing',
        action='store_true',
        help='This does Data Preprocessing. The original dataset `OCDetect_Export` must be present in the data folder.\nDo not use if the data is already preprocessed.\n Read the README.md file for more information.') 
    

    args = arg_parser.parse_args()


    # Set current working directory
    cl.set_main_path(os.getcwd())
    
    # Check if config file is provided
    if args.config == 'None':
        print("Please provide the config file path --config <path>")
        raise ValueError("Config file not provided.")
    
    # Check if method is provided
    if args.method == 'None':
        print("Please provide the method to use --m <method>. Use \'ml\' or \'dl\'")
        raise ValueError("Method not provided.")
    
  
        
    # Load the config file
    cl.load_config(args.config, task_type=args.task_type, architecture=args.architecture)
    
    if cl.config:
        print('Config file loaded successfully.')
    else:
        print('Config file is empty.')
        raise ValueError("Config file is empty. Provide a valid config file. Example: config/default.yaml")

        
    # Create Logger
    Logger.info("Logger created.")

    
    # Task Type
    task_type = TaskType(cl.config.dataset.task_type)
    Logger.info(f"Task type: {task_type}")
    
    
    # Perform data cleaning and preprocessing
    if args.data_preprocessing:
        # Perform data cleaning and preprocessing
        Logger.info("For ML, Performing data cleaning and preprocessing...")
        cln.clean_all()
        Logger.info("Data cleaning and preprocessing completed for ML.")
        
        Logger.info("For DL, Performing data cleaning and preprocessing...")
        # Prepare dataset
        prc.make_datasets("OCDetect_sep_380")
        Logger.info("Data cleaning and preprocessing completed for DL.")
    

    
    # Set GPUs
    if cl.config.world_size == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    elif cl.config.world_size == 4:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        

    # Task Type Information
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
    log_config(cl.config_dict)

    # Datasets selection
    if args.method == 'dl':
        cl.config.dataset.name = "OCDetect_sep_380"
        dl.train()
    elif args.method == 'ml':
        cl.config.dataset.name = "processed"
        #cl.config.train.cross_validation.name = "losocv"
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

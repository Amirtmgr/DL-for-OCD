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

def main():
    # Parse the arguments
    arg_parser = argparse.ArgumentParser(description="Path for the config file.")
    
    # Path for yaml config file
    arg_parser.add_argument(
        'config',
        default='None',
        help='The Configuration file in yaml file format.')
    
    args = arg_parser.parse_args()

    # Set current working directory
    cl.set_main_path(os.getcwd())
    # Load the config file
    cl.load_config(args.config)

    if cl.config:
        print('Config file loaded successfully.')
        for key, value in cl.config.items():
            print(key, ':', value)
    else:
        print('Config file is empty.')
    
    print(cl.config.train.device)
    print(cl.config.models_path)
    #cl.export_to_environment(["main_path"])
    #print(os.environ['main_path'])

    # Create Logger
    Logger.info("Logger created.")

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
        
    dl.train()

    # Visualize all plots
    #vis.show()

    
if __name__ == '__main__':
    main()

# Description: Main file to run the project.
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

# Ignore warnings
import warnings
warnings.filterwarnings('always')

# Import the necessary packages
import os
import argparse
from pathlib import Path
from src.utils.config_loader import config_loader as cl
from src.helper.logger import Logger
import src.dl_pipeline.dl as dl
import src.helper.plotter as plot


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

    # Perform DL pipeline
    dl.train()
 
if __name__ == '__main__':
    main()

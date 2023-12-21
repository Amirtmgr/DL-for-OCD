# ------------------------------------------------------------------------
# Description: Logger Module
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import os
import logging
import datetime
import src.helper.directory_manager as dm
from src.utils.config_loader import config_loader as cl
from abc import ABCMeta, abstractmethod

import logging

class Logger:
    logger = None

    @classmethod
    def initialize_logger(cls):
        # Create a logger and set the logging level
        cls.logger = logging.getLogger("OCD_DL_Logger")
        cls.logger.setLevel(logging.DEBUG)

        # Get log file name
        filename = cl.config.folder + ".txt"
        print("Log file: ", filename)
        # Create folder
        
        path = dm.create_folder(cl.config.folder, dm.FolderType.logs)
        full_path = os.path.join(path, filename)
        print("Log file path: ", full_path)
        
        # Create a logger
        cls.logger = logging.getLogger('MyLogger')
        cls.logger.setLevel(logging.DEBUG)

        # Create a file handler and set the level to DEBUG
        file_handler = logging.FileHandler(full_path)
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set the formatter for the file handler
        #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        cls.logger.addHandler(file_handler)

    @classmethod
    def debug(cls, message):
        print(f"Debug: message")
        cls.logger.debug(message)
    
    @classmethod
    def info(cls, message):
        if cls.logger is None:
            cls.initialize_logger()
        print(message)
        cls.logger.info(message)
    
    @classmethod
    def warning(cls, message):
        print(f"Warning: {message}")
        cls.logger.warning(message)

    @classmethod
    def error(cls, message):
        print(f"Error: {message}")
        cls.logger.error(message)

    @classmethod
    def critical(cls, message):
        print(f"Critical Error: {message}")
        cls.logger.critical(message)

    @classmethod
    def log(cls, message):
        print(message)      
        cls.logger.info(message)

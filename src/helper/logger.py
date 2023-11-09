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
        cls.logger.debug(message)
    
    @classmethod
    def info(cls, message):
        if cls.logger is None:
            cls.initialize_logger()
        cls.logger.info(message)
    
    @classmethod
    def warning(cls, message):
        cls.logger.warning(message)

    @classmethod
    def error(cls, message):
        cls.logger.error(message)

    @classmethod
    def critical(cls, message):
        cls.logger.critical(message)

    @classmethod
    def log(cls, message):        
        cls.logger.info(message)

    
"""        
# Todo: Review Class
class Logger(metaclass=ABCMeta):
    def __init__(self):
      
        
        # Create a logger and set the logging level
        self.logger = logging.getLogger("ocd_ml_logger")
        self.logger.setLevel(logging.DEBUG)

        # Get log file name
        filename = cl.config.folder + ".txt"
        print("Log file: ", filename)
        # Create folder
        path = dm.create_folder(cl.config.folder, dm.FolderType.logs)
        
        full_path = os.path.join(path, filename)  

        # Create a file handler to write logs to the specified file
        file_handler = logging.FileHandler(full_path)
        file_handler.setLevel(logging.DEBUG)

        # Create a console handler to print logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter to format log messages
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Set the formatter for the handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def __generate_log_file_name(self):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        log_file_name = f"log_{formatted_datetime}.txt"
        return log_file_name
"""
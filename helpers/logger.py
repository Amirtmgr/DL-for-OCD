import logging
import datetime
import directory_manager as dm
import os
    
class Logger:
    def __init__(self, folder="logs/"):
        # Get log file name
        log_file = self.__generate_log_file_name()
        
        # Create folder
        dm.create_folder(folder)
        
        # Path of log file
        path = dm.get_data_dir(folder+log_file)
        
        # Create a logger and set the logging level
        self.logger = logging.getLogger("ocd_ml_logger")
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler to write logs to the specified file
        file_handler = logging.FileHandler(path)
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


# Logger
logger = Logger()
logger.info("Logger Created.")
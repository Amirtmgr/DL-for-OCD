# ------------------------------------------------------------------------
# Description: State 
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import os
import logging


# State class
class State:

    def __init__(self):
        self.file_name = None
        self.path = None
        self.epoch = None
        self.model = None
        self.optimizer = None
        self.train_metrics = None
        self.val_metrics = None
        self.lr_scheduler = None

    def set_file_name(self, file_name):
        self.file_name = file_name

    def get_file_name(self):
        return self.file_name
    
    def set_path(self, path):
        self.path = path
    
    def get_path(self):
        return self.path
    
    def info(self):
        print("File name: ", self.file_name)
        print("Path: ", self.path)
        print("Epoch: ", self.epoch)
        print("Model: ", self.model)
        print("Optimizer: ", self.optimizer)
        print("Train metrics: ", self.train_metrics)
        print("Validation metrics: ", self.val_metrics)
        print("Learning rate scheduler: ", self.lr_scheduler)


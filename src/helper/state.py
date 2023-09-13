# ------------------------------------------------------------------------
# Description: State 
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import os
import logging
import src.helper.plotter as plot   

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
        self.train_metrics_arr = None
        self.val_metrics_arr   = None

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


    def get_list_of(self, item='loss', phase='train'):
        """Method to get list of items from train or val metrics array.

        Args:
            item (str, optional): Item type to get. Defaults to 'loss'. Loss, accuracy, f1_score, recall_score, precision_score, specificity_score, jaccard_score.
            phase (str, optional): Item of the phase either "train" or "val" Defaults to 'train'.

        Returns:
            list: List of items. If item is invalid, returns None.
        """

        if self.train_metrics_arr is None:
            logging.warning("Train metrics array is None. Returning None.")
            return None
        
        arr = []

        if phase == 'train':
            metrices = self.train_metrics_arr
        elif phase == 'val':
            metrices  = self.val_metrics_arr
        

        for metric in metrices:
            
            if item == "loss":
                arr.append(metric.loss)
            elif item == "accuracy":
                arr.append(metric.accuracy)
            elif item == "f1_score":
                arr.append(metric.f1_score)
            elif item == "recall_score":
                arr.append(metric.recall_score)
            elif item ==  "precision_score":
                arr.append(metric.precision_score)
            elif item == "specificity_score":
                arr.append(metric.specificity_score)
            elif item == "jaccard_score":
                arr.append(metric.jaccard_score)
            elif item == "confusion_matrix":
                arr.append(metric.confusion_matrix)
            else:
                logging.warning("Invalid item. Returning None.")
                return None

        return arr
    
    def plot_losses(self):
        title = "Train Losses vs Validataion Losses"
        x_label = "Epochs"
        y_label = "Losses"
        legend_labels = ["Train Losses", "Validation Losses"]
        train_losses = self.get_list_of(item='loss', phase='train')
        val_losses = self.get_list_of(item='loss', phase='val')
        data_lists = [train_losses, val_losses]
        plot.arrays(data_lists, title, x_label, y_label, legend_labels)
    
    def plot_f1_scores(self):
        title = "Train F1 Scores vs Validataion F1 Scores"
        x_label = "Epochs"
        y_label = "F1 Scores"
        legend_labels = ["Train F1 Scores", "Validation F1 Scores"]
        train_f1_scores = self.get_list_of(item='f1_score', phase='train')
        val_f1_scores = self.get_list_of(item='f1_score', phase='val')
        data_lists = [train_f1_scores, val_f1_scores]
        plot.arrays(data_lists, title, x_label, y_label, legend_labels)
        

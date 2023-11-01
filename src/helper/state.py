# ------------------------------------------------------------------------
# Description: State 
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import os
import numpy as np
import src.helper.plotter as pl   
from src.helper.logger import Logger
from collections import Counter

# State class
class State:

    def __init__(self):
        self.file_name = None
        self.path = None
        self.best_epoch = None
        self.best_model = None
        self.best_criterion_weight = None
        self.best_optimizer = None
        self.best_train_metrics = None
        self.best_val_metrics = None
        self.best_lr_scheduler = None
        self.train_metrics_arr = None
        self.val_metrics_arr   = None
        self.scaler = None
        self.best_fold = None

    def set_file_name(self, file_name):
        self.file_name = file_name

    def get_file_name(self):
        return self.file_name
    
    def set_path(self, path):
        self.path = path
    
    def get_path(self):
        return self.path
    
    def info(self):
        Logger.info(f"File name: {self.file_name}")
        Logger.info(f"Path: {self.path}")
        Logger.info(f"Best epoch: {self.best_epoch}")
        Logger.info(f"Best model: {self.best_model}")
        Logger.info(f"Best criterion weight: {self.best_criterion_weight}")
        Logger.info(f"Best optimizer: {self.best_optimizer}")
        Logger.info(f"Best train metrics: {self.best_train_metrics.f1_score}")
        Logger.info(f"Best val metrics: {self.best_val_metrics.f1_score}")
        Logger.info(f"Avg Train F1-Score:{self.get_mean(item='f1_score', phase='train')}")
        Logger.info(f"Avg Val F1-Score:{self.get_mean(item='f1_score', phase='val')}")
        Logger.info(f"Avg Train Best Optimal Threshold:{self.get_mean(item='best_threshold', phase='train')}")
        Logger.info(f"Avg Val Best Optimal Threshold:{self.get_mean(item='best_threshold', phase='val')}")
        Logger.info(f"Best Train Threshold:{self.get_best_threshold(phase='train')}")
        Logger.info(f"Best Val Threshold:{self.get_best_threshold(phase='val')}")

        if self.best_lr_scheduler:
            if hasattr(self.best_lr_scheduler, 'last_lr'):
                Logger.info(f"Best last_lr: {self.best_lr_scheduler.last_lr}")
        #Logger.info(f"Train metrics array: {self.train_metrics_arr}")
        #Logger.info(f"Val metrics array: {self.val_metrics_arr}")

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
            elif item == "best_threshold":
                arr.append(metric.best_threshold)
            else:
                logging.warning("Invalid item. Returning None.")
                return None

        return arr
    
    def plot_losses(self, title=""):
        title = "Train Losses vs Validataion Losses" + title
        x_label = "Epochs"
        y_label = "Losses"
        legend_labels = ["Train Losses", "Validation Losses"]
        train_losses = self.get_list_of(item='loss', phase='train')
        val_losses = self.get_list_of(item='loss', phase='val')
        data_lists = [train_losses, val_losses]
        #plot.arrays(data_lists, title, x_label, y_label, legend_labels)
        pl.plotly_arrays(data_lists, title, x_label, y_label, legend_labels)
    
    def plot_f1_scores(self, title=""):
        title = "Train F1 Scores vs Validataion F1 Scores"+ title
        x_label = "Epochs"
        y_label = "F1 Scores"
        legend_labels = ["Train F1 Scores", "Validation F1 Scores"]
        train_f1_scores = self.get_list_of(item='f1_score', phase='train')
        val_f1_scores = self.get_list_of(item='f1_score', phase='val')
        data_lists = [train_f1_scores, val_f1_scores]
        #plot.arrays(data_lists, title, x_label, y_label, legend_labels)
        pl.plotly_arrays(data_lists, title, x_label, y_label, legend_labels)
        

    def get_mean(self, item='loss', phase='train'):
        """Method to get mean of items from train or val metrics array.

        Args:
            item (str, optional): Item type to get. Defaults to 'loss'. Loss, accuracy, f1_score, recall_score, precision_score, specificity_score, jaccard_score.
            phase (str, optional): Item of the phase either "train" or "val" Defaults to 'train'.

        Returns:
            float: Mean of items. If item is invalid, returns None.
        """

        arr = self.get_list_of(item, phase)

        if arr is None:
            return None

        avg = np.mean(arr)

        return avg
    
    def get_best_threshold(self, phase='train'):
        """Method to get best threshold from train or val metrics array.

        Args:
            phase (str, optional): Item of the phase either "train" or "val" Defaults to 'train'.

        Returns:
            float: Best threshold. If item is invalid, returns None.
        """

        arr = self.get_list_of(item='best_threshold', phase=phase)

        if arr is None:
            return None

        counts = Counter(arr)
        #print(counts)
        # Find the maximum count and corresponding value
        max_count = max(counts.values())
        most_common_value = [key for key, value in counts.items() if value == max_count][0]

        return most_common_value

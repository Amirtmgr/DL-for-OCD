# ------------------------------------------------------------------------
# Description: Metrics Class
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import numpy as np
import warnings
from collections import Counter
from tabulate import tabulate

from src.utils.config_loader import config_loader as cl

from src.helper.logger import Logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    jaccard_score
)

class Metrics:
    def __init__(self, epoch, is_binary, labels=[0 ,1 , 2], averaging='macro'):
        self.phase = ""
        self.epoch = epoch
        self.is_binary = is_binary
        self.loss = None
        self.averaging = averaging
        self.f1_score = 0.0
        self.recall_score = 0.0
        self.precision_score = 0.0
        self.specificity_score = 0.0
        self.confusion_matrix = 0.0
        self.jaccard_score = 0.0
        self.accuracy = 0.0
        self.y_true = None
        self.y_pred = None
        self.labels = [0, 1] if is_binary else [0, 1, 2]
        self.zero_division_warn = False
        self.classification_report = None

        try:
            
            if cl.config.metrics.zero_division == 'nan':
                self.zero_division = np.nan
            else:
                self.zero_division = cl.config.metrics.zero_division

        except Exception as e:
            Logger.warning(f"Phase:{self.phase} Zero division not found in config file. {str(e)} Set to 'warn'.")
            self.zero_division = 'warn'
        
    def calculate_metrics(self):
        y_true = self.y_true
        y_pred = self.y_pred
        Logger.info(f"Phase {self.phase} | y_true: {np.unique(y_true)} | y_pred: {np.unique(y_pred)}")
        Logger.info(f"Phase {self.phase} | y_true counts: {Counter(y_true)} | y_pred shape: {Counter(y_pred)}")
        print(f"Epoch {self.epoch + 1} Phase {self.phase} | y_true: {np.unique(y_true)} | y_pred: {np.unique(y_pred)}")
        print(f"Epoch {self.epoch + 1} Phase {self.phase} | y_true counts: {Counter(y_true)} | y_pred shape: {Counter(y_pred)}")

        # Catch ZeroDivision UserWarning
        with warnings.catch_warnings(record=True) as warn_list:

            try:
                # Accuracy
                self.accuracy = accuracy_score(y_true, y_pred)

                # F1 Score
                self.f1_score = f1_score(y_true, y_pred, labels=self.labels, average=self.averaging, zero_division=self.zero_division)

                # Recall
                self.recall_score = recall_score(y_true, y_pred, labels=self.labels, average=self.averaging, zero_division=self.zero_division)

                # Precision
                self.precision_score = precision_score(y_true, y_pred, labels=self.labels, average=self.averaging, zero_division=self.zero_division)

                # Confusion Matrix
                self.confusion_matrix = confusion_matrix(y_true, y_pred, labels=self.labels)

                # Jaccard Score
                self.jaccard_score = jaccard_score(y_true, y_pred, labels=self.labels, average=self.averaging, zero_division=self.zero_division)
                
                # Classification Report
                self.classification_report = classification_report(y_true, y_pred, labels=self.labels, zero_division=self.zero_division)
                #Logger.info(self.classification_report)
                print(f"Accuracy: {self.accuracy:.2f}")
                print(f"F1 Score: {self.f1_score:.2f}")
                print(f"Recall: {self.recall_score:.2f}")
                print(f"Precision: {self.precision_score:.2f}")
                print(f"Jaccard Score: {self.jaccard_score:.2f}")
                
                self.print_cm()
                print(f"Report:\n {self.classification_report}")

                # Specificity
                if len(self.labels) == 2:
                    tn, fp, fn, tp = self.confusion_matrix.ravel()
                    self.specificity_score = tn / (tn + fp) if (tn + fp) != 0 else 0.0
                    print(f"Specificity: {self.specificity_score:.2f}")
                
            except Warning as w:
                self.zero_division_warn = True
                Logger.warning(f"Phase: {self.phase} | An warning occurred: {str(w)}")

            except Exception as e:
                Logger.error(f"Phase: {self.phase} | An error occurred: {str(e)}")
                
            except ZeroDivisionError as e:
                Logger.warning(f"Phase : {self.phase} | An error occurred: {str(e)}")
                
            finally:
                pass
                
    def get_f1_score(self):
        return self.f1_score

    def get_recall_score(self):
        return self.recall_score

    def get_precision_score(self):
        return self.precision_score

    def get_specificity_score(self):
        return self.specificity_score

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_jaccard_score(self):
        return self.jaccard_score
    
    def set_loss(self, loss):
        self.loss = loss

    def info(self):
        if not self.zero_division_warn:
            msg = f"Phase {self.phase} : Metrics: F1_Score: {self.f1_score} | Recall: {self.recall_score} | Precision: {self.precision_score} | Specificity: {self.specificity_score} | Jaccard: {self.jaccard_score} | Accuracy: {self.accuracy}"
        else:
            msg = f"[Zero_Division Warning] Phase {self.phase} : Metrics: F1_Score: {self.f1_score} | Recall: {self.recall_score} | Precision: {self.precision_score} | Specificity: {self.specificity_score} | Jaccard: {self.jaccard_score} | Accuracy: {self.accuracy}"
        Logger.info(msg)
    
    # To do:
    def ravel_confusion_matrix(self):
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        return tn, fp, fn, tp

    
    # Print Confusion Matrix
    def print_cm(self):
        """
        Print a confusion matrix in a beautiful table format.

        Args:
            labels (list): List of class labels.
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
        """
        labels = cl.config.dataset.labels

        # Calculate the confusion matrix
        cm = self.confusion_matrix

        # Create a table with labels
        table = [[label] + [str(value) if value is not None else '' for value in row] for label, row in zip(labels, cm)]

        # Add headers for columns and rows
        headers = [""] + labels
        table.insert(0, headers)

        # Print the confusion matrix as a table
        print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


        
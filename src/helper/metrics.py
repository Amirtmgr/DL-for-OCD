# ------------------------------------------------------------------------
# Description: Metrics Class
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import numpy as np

from src.utils.config_loader import config_loader as cl

from src.helper.logger import Logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    jaccard_score
)

class Metrics:
    def __init__(self, averaging='macro'):
        self.loss = None
        self.averaging = averaging
        self.f1_score = None
        self.recall_score = None
        self.precision_score = None
        self.specificity_score = None
        self.confusion_matrix = None
        self.jaccard_score = None
        self.accuracy = None
        self.y_true = None
        self.y_pred = None
        try:
            
            if cl.config.metrics.zero_division == 'nan':
                self.zero_division = np.nan
            else:
                self.zero_division = cl.config.metrics.zero_division

        except Exception as e:
            Logger.warning(f"Zero division not found in config file. {str(e)} Set to 'warn'.")
            self.zero_division = 'warn'
        
    def calculate_metrics(self):
        y_true = self.y_true
        y_pred = self.y_pred
        Logger.info(f"y_true: {np.unique(y_true)} | y_pred: {np.unique(y_pred)}")
        
        try:
            # Accuracy
            self.accuracy = accuracy_score(y_true, y_pred)

            # F1 Score
            self.f1_score = f1_score(y_true, y_pred, average=self.averaging, zero_division=self.zero_division) 

            # Recall
            self.recall_score = recall_score(y_true, y_pred, average=self.averaging, zero_division=self.zero_division)

            # Precision
            self.precision_score = precision_score(y_true, y_pred, average=self.averaging, zero_division=self.zero_division)

            # Confusion Matrix
            self.confusion_matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2])

            # Jaccard Score
            self.jaccard_score = jaccard_score(y_true, y_pred, average=self.averaging, zero_division=self.zero_division)

            # Specificity
            self.specificity_score = ((1 + self.precision_score) * self.recall_score) / (self.precision_score + self.recall_score)

        except Exception as e:
            #Logger.error(f"An error occurred: {str(e)}")
            print(f"An error occurred: {str(e)}")

        except ZeroDivisionError as e:
            Logger.error(f"An error occurred: {str(e)}")
            Logger.warning(f"An error occurred: {str(e)}")
            Logger.info(f"y_target: {np.unique(y_true)} | y_pred: {np.unique(y_pred)}")
            
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

    def info(self, title=""):
        msg = f"{title} Metrics: F1_Score: {self.f1_score} | Recall: {self.recall_score} | Precision: {self.precision_score} | Specificity: {self.specificity_score} | Jaccard: {self.jaccard_score} | Accuracy: {self.accuracy}"
        Logger.info(msg)
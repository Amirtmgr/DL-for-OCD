# ------------------------------------------------------------------------
# Description: Metrics Class
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

from src.helper.logger import Logger
from sklearn.metrics import (
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

    def calculate_metrics(self, y_true, y_pred, pos_label=None):
        try:
            # F1 Score
            self.f1_score = f1_score(y_true, y_pred, average=self.averaging, pos_label=pos_label) # type: ignore
            
            # Recall
            self.recall_score = recall_score(y_true, y_pred, average=self.averaging, pos_label=pos_label) # type: ignore
            
            # Precision
            self.precision_score = precision_score(y_true, y_pred, average=self.averaging, pos_label=pos_label) # type: ignore
            
            # Confusion Matrix
            self.confusion_matrix = confusion_matrix(y_true, y_pred)
            
            # Jaccard Score
            self.jaccard_score = jaccard_score(y_true, y_pred, average=self.averaging, pos_label=pos_label) # type: ignore 
            
            # Specificity
            tn, fp, fn, tp = self.confusion_matrix.ravel()
            self.specificity_score = tn / (tn + fp)
            
        except Exception as e:
            #Logger.error(f"An error occurred: {str(e)}")
            print(f"An error occurred: {str(e)}")

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
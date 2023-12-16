# ------------------------------------------------------------------------
# Description: Metrics Class
# Author: Amir Thapa Magar
# Email: amir.thapamagar(at)student.uni-siegen.de
# ------------------------------------------------------------------------

import numpy as np
import warnings
from collections import Counter
from tabulate import tabulate
from src.helper import plotter as pl
from src.utils.config_loader import config_loader as cl

from src.helper.cf_matrix import make_confusion_matrix
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
        self.labels = [0, 1] if cl.config.dataset.task_type < 3 else [0, 1, 2]
        self.zero_division_warn = False
        self.classification_report = None
        self.outputs = None
        self.best_threshold = 0.5
        #self.s1_score = 0.0

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
                self.accuracy = round(accuracy_score(y_true, y_pred),2)

                # F1 Score
                self.f1_score = round(f1_score(y_true, y_pred, labels=self.labels, average=self.averaging, zero_division=self.zero_division),2)

                # Recall
                self.recall_score = round(recall_score(y_true, y_pred, labels=self.labels, zero_division=self.zero_division),2)

                # Precision
                self.precision_score = round(precision_score(y_true, y_pred, labels=self.labels, zero_division=self.zero_division),2)
                
                # Confusion Matrix
                self.confusion_matrix = confusion_matrix(y_true, y_pred, labels=self.labels)

                # Jaccard Score
                self.jaccard_score = round(jaccard_score(y_true, y_pred, labels=self.labels, average=self.averaging, zero_division=self.zero_division),2)
                
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
                Logger.info(f"Report:\n {self.classification_report}")
                
                # Specificity
                if len(self.labels) == 2:
                    tn, fp, fn, tp = self.confusion_matrix.ravel()
                    self.specificity_score = round(tn / (tn + fp), 2) if (tn + fp) != 0 else 0.0
                    print(f"Specificity: {self.specificity_score:.2f}")

                    #self.s1_score = round(2 * ((self.specificity_score * self.recall_score) / (self.specificity_score + self.recall_score)), 2)
                    #print(f"S1 Score: {self.s1_score:.2f}")

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
        Logger.info(f"Epoch: {self.epoch+1} Confusion Matrix:")
        Logger.info("\n"+tabulate(table, headers="firstrow", tablefmt="fancy_grid"))


    def save_cm(self, info=""):
        # cf = self.confusion_matrix
        # categories = cl.config.dataset.labels
        # stats = f"\n\nF1-Score: {self.f1_score:.2f}\nRecall: {self.recall_score:.2f}\nPrecision: {self.precision_score:.2f}\nJaccard: {self.jaccard_score:.2f}"
        # pl.plot_cm(cf, categories, info, stats=None, save_fig=True)

        categories = cl.config.dataset.labels
        cm = self.confusion_matrix
        if self.is_binary:
            labels = ["True Neg","False Pos","False Neg","True Pos"]
            title=f"{categories[0]} vs {categories[1]}"
            
            make_confusion_matrix(cm, 
                        group_names=labels,
                        categories=categories,
                        title=title + info,
                        save=True,
                        )
        else:
            make_confusion_matrix(cm, 
                        categories=categories,
                        title="Multi-class Confusion Matrix" + info,
                        save=True,
                        )
            

    # Compute optimal threshold
    def compute_optim_threshold(self, metric='f1', num_thresholds=100):

        if not self.is_binary:
            return self.best_threshold

        thresholds = np.linspace(0, 1, num_thresholds)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            predicted_labels = (self.outputs > threshold).astype(int)
            if metric == 'f1':
                #score = f1_score(true_labels, predicted_labels)
                score = f1_score(self.y_true, predicted_labels, labels=self.labels, average=self.averaging, zero_division=self.zero_division)

            # You can add other metrics like accuracy, precision, and recall here

            if score > best_score:
                best_score = score
                best_threshold = threshold
                self.best_threshold = best_threshold
                #Logger.info(f"Best threshold: {best_threshold:.2f} | Best {metric} score: {best_score:.2f}")
                
        return best_threshold

    
    def plot_cm(self, info=""):
        
        cf = self.confusion_matrix
        blanks = ['' for i in range(cf.size)]
        categories = cl.config.dataset.labels

        if self.is_binary:
            group_names = ["True Neg","False Pos","False Neg","True Pos"]
            title = f"{categories[0]} vs {categories[1]} " + info
        else:
            group_names = None
            title = "Multiclass Confusion Matrix" + info

        if group_names and len(group_names)==cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
        
        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


        sns.heatmap(cf,annot=box_labels,fmt="g",cbar=True,xticklabels=True,yticklabels=True)

        stats_text = "\n\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                self.precision_score, self.recall_score, self.f1_score)

        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
        plt.title(title)
        pl.save_plot(plt, title)

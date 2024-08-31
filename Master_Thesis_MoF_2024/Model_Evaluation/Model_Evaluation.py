# Importing External Libraries
import pandas as pd

# Importing Internal Libraries
from Model_Evaluation.methods.calculate_accuracy import calculate_accuracy
from Model_Evaluation.methods.calculate_confusion_matrix import\
    calculate_confusion_matrix
from Model_Evaluation.methods.calculate_precision_recall_f1 import\
    calculate_precision_recall_f1
from Model_Evaluation.methods.plot_roc_auc import plot_roc_auc


"""
Setting up class to prepare data for analysis
"""
class Model_Evaluation:
    """
    Class dedicated to evaluation ML algorithms

    
    ***Methods***
    1. calculate_accuracy: Calculates accuracy measure for oos forecasting
    
    2. calculate_precision_recall_f1: Calculates precision, recall & F1 measures for oos forecasting
    
    3. plot_roc_auc: Plots ROC-AUC for oos forecasting
    
    4. calculate_confusion_matrix: Calculates & Plots confusion matrix for oos forecasting
    """
    def __init__(self, model):
        
        self.model = model
        
        return
    
    def calculate_accuracy(self, y_test: pd.Series(), y_pred: pd.Series()):
        
        return calculate_accuracy(y_test, y_pred)
    
    def calculate_precision_recall_f1(self, y_test: pd.Series(), y_pred: pd.Series()):
        
        return calculate_precision_recall_f1(y_test, y_pred)
        
    def plot_roc_auc(self, y_test: pd.Series(), y_pred: pd.Series(), title:str()):
        
        return plot_roc_auc(y_test, y_pred, title)
        
    def calculate_confusion_matrix(self, y_test: pd.Series(), y_pred: pd.Series(),\
                                   title:str()):
        
        return calculate_confusion_matrix(y_test, y_pred, title)
        

    
    
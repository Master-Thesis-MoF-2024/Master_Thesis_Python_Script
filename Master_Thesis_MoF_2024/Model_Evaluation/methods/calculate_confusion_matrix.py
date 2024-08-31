# Importing external libraries
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


def calculate_confusion_matrix(y_test: pd.Series(), y_pred: pd.Series(),\
                               title:str()):
    
    cm = confusion_matrix(y_test, y_pred) # Calculate Confusion Matrix
    
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues") # Showing confusion matrix

    return print("Confusion Matrix:\n", cm)

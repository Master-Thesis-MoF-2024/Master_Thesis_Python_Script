# Importing External Libraries
from sklearn.metrics import accuracy_score
import pandas as pd

def calculate_accuracy(y_test: pd.Series(), y_pred: pd.Series()):
    
    accuracy = accuracy_score(y_test, y_pred) # Calculating accuracy

    
    return accuracy

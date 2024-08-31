from sklearn.metrics import classification_report
import pandas as pd



def calculate_precision_recall_f1(y_test: pd.Series(), y_pred: pd.Series()):

    # Alternatively, you can use classification_report to get all metrics at once
    report = classification_report(y_test, y_pred)
    
    
    return print("\nClassification Report:\n", report)
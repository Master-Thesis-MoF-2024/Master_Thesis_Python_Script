# Importing External Libraries
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd

def plot_roc_auc(y_test: pd.Series(), y_pred: pd.Series(), title:str()):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random guess)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title}: Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc='lower right')
    plt.grid(True)
    
    
    return plt.show()

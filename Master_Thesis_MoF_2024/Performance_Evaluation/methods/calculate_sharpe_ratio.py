# Importing External Libraries
import pandas as pd

# Importing Internal libraries
from Performance_Evaluation.methods.calculate_mean import calculate_mean
from Performance_Evaluation.methods.calculate_std import calculate_std

def calculate_sharpe_ratio(performance: pd.Series, rf:float = 0.024):
    
    rp = calculate_mean(performance)
    stdp = calculate_std(performance)
    
    ratio = (rp - rf) / stdp
        
    
    return ratio


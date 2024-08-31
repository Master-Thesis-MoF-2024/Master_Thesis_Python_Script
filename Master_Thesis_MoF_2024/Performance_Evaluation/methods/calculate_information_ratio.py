# Importing External libraries
import pandas as pd
import numpy as np

# Importing Internal libraries
from Performance_Evaluation.methods.calculate_mean import calculate_mean

def calculate_information_ratio(performance_p: pd.Series, performance_m: pd.Series):
    
    rtn_p = np.log(performance_p / performance_p.shift(1)).fillna(0)
    rtn_m = np.log(performance_m / performance_m.shift(1)).fillna(0)
    
    mean_p = calculate_mean(performance_p)
    mean_m = calculate_mean(performance_m)
    
    TE = np.std(rtn_p - rtn_m)
    
    ratio = (mean_p - mean_m) / TE
    
    return ratio



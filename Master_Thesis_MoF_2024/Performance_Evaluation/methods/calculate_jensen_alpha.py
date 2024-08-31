# Importing External libraries
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Importing Internal libraries
from Performance_Evaluation.methods.calculate_mean import calculate_mean

def calculate_jensen_alpha(performance_p: pd.Series, performance_m: pd.Series,\
                           rf = 0.024):
    
    rtn_p = np.log(performance_p / performance_p.shift(1)).fillna(0)
    rtn_m = np.log(performance_m / performance_m.shift(1)).fillna(0)
    
    mean_p = calculate_mean(performance_p)
    mean_m = calculate_mean(performance_m)
    
    slope, intercept, r_value, p_value, std_err = linregress(rtn_m, rtn_p)
    
    beta = slope
    
    alpha = mean_p - (rf + beta*(mean_m - rf))
    
    return alpha


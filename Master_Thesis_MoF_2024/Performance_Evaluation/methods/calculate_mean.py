# Importing External libraries
import pandas as pd
import numpy as np

def calculate_mean(performance: pd.Series):
    
    rtn = np.log(performance / performance.shift(1)).fillna(0)
        
    mean = np.mean(rtn)
    
    # Annualizing
    mean = ((1+mean)**12)-1
    
    return mean


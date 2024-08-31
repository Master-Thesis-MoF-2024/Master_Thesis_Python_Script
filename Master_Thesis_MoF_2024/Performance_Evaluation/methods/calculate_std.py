# Importing External libraries
import pandas as pd
import numpy as np

def calculate_std(performance: pd.Series):
    
    rtn = np.log(performance / performance.shift(1)).fillna(0)
        
    std = np.std(rtn)
    
    return std

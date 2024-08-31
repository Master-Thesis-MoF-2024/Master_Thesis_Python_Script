# Importing External Libraries
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize


def clean_data(df1: dict):
    
    for i in df1.keys():
        
        
        df1[i].replace(0, np.nan, inplace = True) # Replacing 0 with NaN

        for j in df1[i].columns:  
        
            winsorize(df1[i][j], limits= [0.05, 0.05], inplace=True) # Removing outliers    
    
        
        df1[i] = df1[i].apply(lambda x: x.fillna(x.mean()), axis=0) # Substituting NaNs with mean

     
    return df1

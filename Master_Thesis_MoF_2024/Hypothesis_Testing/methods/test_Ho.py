# Importing external libraries
import numpy as np
import pandas as pd

# Importing Internal libraries
from Performance_Evaluation.methods.calculate_mean import calculate_mean
from Performance_Evaluation.methods.calculate_std import calculate_std

def test_Ho(port1: pd.Series, port2: pd.Series):
    
    sample_size = len(port1)
    sample_mean = calculate_mean(port1)
    sample_std = calculate_std(port1)
    population_mean_under_Ho = calculate_mean(port2)
    
    
    Z = (sample_mean-population_mean_under_Ho)/(sample_std/(np.sqrt(sample_size)))
    
    z_value = 1.645     
    
    if Z > z_value:
        
        return print(f"We reject the null hypothesis {Z:.2f} > {z_value}")
            
    else:
        return print(f"We fail to reject the null hypothesis {Z:.2f} < {z_value}")
    
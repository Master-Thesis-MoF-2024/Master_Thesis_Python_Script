# Importing External libraries
import pandas as pd
import numpy as np


def calculate_max_drawdown(performance: pd.Series()):
    
    rtn = np.log(performance / performance.shift(1)).fillna(0)
    
    cum = (1 + rtn).cumprod()

    # Calculate drawdown
    draw = cum / cum.cummax() - 1
    
    # Calculate maximum drawdown
    max_drawdown_portfolio = draw.min()
    
    return max_drawdown_portfolio


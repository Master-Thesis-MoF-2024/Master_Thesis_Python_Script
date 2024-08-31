# Importing external libraries
import pandas as pd

# Importing internal libraries
from Performance_Evaluation.methods.calculate_mean import calculate_mean
from Performance_Evaluation.methods.calculate_std import calculate_std
from Performance_Evaluation.methods.calculate_sharpe_ratio \
    import calculate_sharpe_ratio
from Performance_Evaluation.methods.calculate_jensen_alpha \
    import calculate_jensen_alpha
from Performance_Evaluation.methods.calculate_information_ratio \
    import calculate_information_ratio
from Performance_Evaluation.methods.calculate_max_drawdown \
    import calculate_max_drawdown
from Performance_Evaluation.methods.plot_performances \
    import plot_performances   

class Performance_Evaluation:
    """
    Class dedicated to evaluation of 2 investing strategies

    
    ***Methods***
    1. calculate_mean: Calculates mean return of portfolio
    
    2. calculate_std: Calculates standard deviation of portfolio
    
    3. calculate_sharpe_ratio: Returns sharpe ratio against a benchmark
    
    4. calculate_information_ratio: Calculates information ratio
    
    5. calculate_max_drawdown: Calculates Max Drawdown for the strategy
    
    6. calculate_jensen_alpha: Calculates Jensen alpha
    
    7. plot_performances: plots performances of 2 strategies
    """
    def __init__(self):
        
        return
    
    def calculate_mean(self, performance: pd.Series):
        
        return calculate_mean(performance)
    
    def calculate_std(self, performance: pd.Series):
        
        return calculate_std(performance)
        
    def calculate_sharpe_ratio(self, performance: pd.Series, rf:float = 0.024):
        
        return calculate_sharpe_ratio(performance, rf = 0.024)
    
    def calculate_information_ratio(self, performance_p: pd.Series, performance_m: pd.Series):
        
        return calculate_information_ratio(performance_p, performance_m)
    
    def calculate_jensen_alpha(self, performance_p: pd.Series, performance_m: pd.Series,\
                               rf = 0.024):
        
        return calculate_jensen_alpha(performance_p, performance_m,rf)
    
    def calculate_max_drawdown(self, performance: pd.Series()):
        
        return calculate_max_drawdown(performance)
    
    
    def plot_performances(self, performance1: pd.Series(), performance2: pd.Series(), names: list()):
        
        return plot_performances(performance1, performance2,names)
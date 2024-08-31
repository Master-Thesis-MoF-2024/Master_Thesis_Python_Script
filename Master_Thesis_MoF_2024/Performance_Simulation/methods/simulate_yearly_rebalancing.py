# Importing external libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing internal libraries
from Training.Training import Training



def simulate_yearly_rebalancing(data_4_training: list(),\
                                ys: list(), data_4_forecasting: list(), \
                                    data_rtn: pd.DataFrame()):
    X = pd.DataFrame()
    y = pd.Series()
    portfolio_performance = pd.Series([100])
    
    
    for i in range(len(data_4_training)): 
        
        X = pd.concat([X, data_4_training[i]], axis=0, \
                         ignore_index=True).reset_index(drop=True)
        
        y = pd.concat([y, ys[i]], axis=0, \
                         ignore_index=True).reset_index(drop=True)
        
        model = Training.train_random_forest(X, y)
        
        prediction = pd.Series(model.predict(data_4_forecasting[i]["Feb"].drop("Outperform", axis=1)))
        
        positions_of_ones = prediction[prediction == 1].index.tolist()
        
        returns = data_rtn[str((2005)+(i+1))].iloc[positions_of_ones].T
        
        for i in range(1, 12):
            
            value_portfolio = portfolio_performance.iloc[-1]
            
    
            stocks = pd.Series([(value_portfolio/len(positions_of_ones))]*len(positions_of_ones)) 
            
            new_value = sum(stocks.reset_index(drop=True) * ((returns.iloc[i].fillna(returns.mean()).reset_index(drop=True))+1))    
            
            
            portfolio_performance = pd.concat([portfolio_performance, pd.Series(new_value)], \
                                              ignore_index=True)

        
    return portfolio_performance
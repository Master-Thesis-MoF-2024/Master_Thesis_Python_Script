# Importing external libraries
import pandas as pd


# Importing internal libraries
from Performance_Simulation.methods.simulate_yearly_rebalancing import\
    simulate_yearly_rebalancing   


class Performance_Simulation:
    """
    Class dedicated to simulation of ML base trading strategies

    
    ***Methods***
    1. simulate_yearly_rebalancing: Simulates ML investment strategy without trading costs
    
    2. simulate_with_trading_costs: Simulates ML investment strategy with trading costs
    """
    def __init__(self):
        
        return
    
    def simulate_yearly_rebalancing(self, data_4_training: list(), ys: list(),\
                                    data_4_forecasting: list(), \
                                        data_rtn: pd.DataFrame()):
        
        return simulate_yearly_rebalancing(data_4_training, ys, data_4_forecasting,\
                                           data_rtn)
   
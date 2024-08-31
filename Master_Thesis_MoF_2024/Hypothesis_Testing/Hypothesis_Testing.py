# Importing external libraries
import pandas as pd

# Importing internal libraries
from Hypothesis_Testing.methods.test_Ho import test_Ho


class Hypothesis_Testing:
    """
    Class dedicated to Hypothesis Testing

    
    ***Methods***
    1. test_Ho: Performs Hypothesis testing either at the 95% or 99% confidence level
    """
    
    def __init__(self):
        
        return
    
    def test_Ho(self, port1: pd.Series, port2: pd.Series):
        
        return test_Ho(port1, port2)
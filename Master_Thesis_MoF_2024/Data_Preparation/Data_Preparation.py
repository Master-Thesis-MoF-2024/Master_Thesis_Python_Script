# Importing External Libraries
import pandas as pd

# Importing Internal Libraries
from Data_Preparation.methods.clean_data import clean_data
from Data_Preparation.methods.calculate_outperformers \
    import calculate_outperformers
from Data_Preparation.methods.create_data_x_training import create_data_x_training


"""
Setting up class to prepare data for analysis
"""
class Data_Preparation:
    """
    Class dedicated to the preparation of data for ML algorithms
    
    ***Methods***
    1. clean_data: Removes NaNs & outliers from dataset
    
    2. calculate_outperformers: Assings a 1 to those stocks which had a selected
    metric which performed above average
    
    3. create_data_x_training: Creates dataset that will be fed to ML algorithm
    """
    def __init__(self, df: pd.DataFrame()):
        self.df = df
        
        return
            
    def clean_data(self):
        
        return clean_data(self.df)
    
    def calculate_outperformers(self, df: pd.DataFrame(), var_name: str()):
        
        return calculate_outperformers(df, var_name)
    
    
    def create_data_x_training(df: dict, y_name: str()):
        
        return create_data_x_training(df, y_name)
    
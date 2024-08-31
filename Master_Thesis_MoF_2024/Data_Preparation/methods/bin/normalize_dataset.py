# Importing External Libraries
import pandas as pd


def normalize_dataset(df: pd.DataFrame()):
    
    new_df = df 
    
    for ws, df in new_df.items():
        
        # Normalizing metrics
        df_normalized = (df - df.mean()) / df.std()

        # Adding to new df updated Worksheet
        new_df[ws] = df_normalized
    
    return new_df
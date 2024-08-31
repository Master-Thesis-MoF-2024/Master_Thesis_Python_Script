# Importing External Libraries
import pandas as pd

def create_data_x_training(df: dict, y_name: str()):
    
    X = pd.DataFrame()
    
    for i in df.keys():
        
        X = pd.concat([X, df[i]], axis=0, ignore_index=True).reset_index(drop=True)
    
    y = X[y_name]
    X = X.drop(y_name, axis=1).apply(lambda x: x.fillna(x.mean()), axis=0)
    
    return X, y
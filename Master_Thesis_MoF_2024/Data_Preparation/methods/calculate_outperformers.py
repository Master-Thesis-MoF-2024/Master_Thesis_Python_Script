# Importing External Libraries
import pandas as pd


def calculate_outperformers(df1: dict, var_name: str()):
    
    for i in df1.keys():
    # Outperformers are those stocks which performed better than the average
        df1[i]["Outperform"] = df1[i][var_name].apply(lambda x: 1 \
                                              if x > df1[i][var_name].mean() else 0)
    
    return df1

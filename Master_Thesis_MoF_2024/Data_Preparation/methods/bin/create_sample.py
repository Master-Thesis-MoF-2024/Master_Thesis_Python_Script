# Importing External Libraries
import pandas as pd


def create_sample(df: pd.DataFrame()):
    sheets_dict = df
    
    df_list = []
    
    for sheet_name, df in sheets_dict.items():
        # Ensure the DataFrame does not have duplicate index or columns
        df = df[~df.index.duplicated(keep='first')]  # Remove any duplicated indices
        df = df.loc[:, ~df.columns.duplicated()]  # Remove any duplicated columns
        
        df = df.stack()  # Convert the DataFrame to a Series with tickers as the innermost level
        df.index.names = ['Date', 'Ticker']  # Name the index levels
        df = df.swaplevel().to_frame(sheet_name)  # Swap index levels so Ticker is the first level and convert back to DataFrame
        
        df_list.append(df)
    
    # Concatenate all the DataFrames along the columns
    result_df = pd.concat(df_list, axis=1)
    
    # Handle any potential duplicate indices after concatenation
    result_df = result_df[~result_df.index.duplicated(keep='first')]
    
    # Stack again to move the worksheet names to a level in the index
    result_df = result_df.stack().unstack(1)
    
    # Reset the index names appropriately
    result_df.index.names = ['Ticker', 'Date']
    result_df.columns.name = 'Worksheet'
    
    # Display the resulting DataFrame
    result_df = result_df.swaplevel().sort_index()  # Swap the levels and sort the index for better readability
    result_df = result_df.swaplevel(0, 1)  # Swap level 0 (Worksheet) and level 1 (Ticker)
    
    # Sort the DataFrame based on the new index structure
    #result_df = result_df.sort_index()
    
    return result_df
    
    
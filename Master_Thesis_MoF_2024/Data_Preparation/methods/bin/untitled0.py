import pandas as pd

# Load the Excel file
file_path = 'C:/Users/debel/Desktop/Augusto_De_Bellis_Master_Thesis/data/trial.xlsx'  # Replace with your file path
excel_data = pd.read_excel(file_path, sheet_name=None)

# Initialize an empty list to store DataFrames for each month
monthly_data = []

# Iterate through each sheet (month) and transform it
for month, df in excel_data.items():
    # Add a 'Month' column to identify the data from each sheet
    df['Month'] = month
    # Melt the DataFrame to get 'Ticker', 'Factor', and 'Value' columns
    df_melted = df.melt(id_vars=['Month'], var_name='Ticker', value_name='Value')
    # Append the melted DataFrame to the list
    monthly_data.append(df_melted)

# Concatenate all the monthly DataFrames into one
combined_data = pd.concat(monthly_data)

# Pivot the combined DataFrame to get the desired structure
df_final = combined_data.pivot_table(index='Month', columns=['Ticker'], values='Value')

# If you want to reorder the index and columns to have Tickers as the top level:
df_final = df_final.sort_index(axis=1, level=0)

# Display the resulting DataFrame
print(df_final)

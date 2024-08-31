# Importing external libraries
import pandas as pd

df_2005 = pd.read_excel("data/datasets/2005.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2005.keys():
    df_2005[i] = df_2005[i][df_2005[i].index.notna()]
    
df_2006 = pd.read_excel("data/datasets/2006.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2006.keys():
    df_2006[i] = df_2006[i][df_2006[i].index.notna()]
    
df_2007 = pd.read_excel("data/datasets/2007.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2007.keys():
    df_2007[i] = df_2007[i][df_2007[i].index.notna()]
    
df_2008 = pd.read_excel("data/datasets/2008.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2008.keys():
    df_2008[i] = df_2008[i][df_2008[i].index.notna()]
    
df_2009 = pd.read_excel("data/datasets/2009.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2009.keys():
    df_2009[i] = df_2009[i][df_2009[i].index.notna()]
    
df_2010 = pd.read_excel("data/datasets/2010.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2010.keys():
    df_2010[i] = df_2010[i][df_2010[i].index.notna()]
    
df_2011 = pd.read_excel("data/datasets/2011.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2011.keys():
    df_2011[i] = df_2011[i][df_2011[i].index.notna()]
    
df_2012 = pd.read_excel("data/datasets/2012.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2012.keys():
    df_2012[i] = df_2012[i][df_2012[i].index.notna()]
    
df_2013 = pd.read_excel("data/datasets/2013.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2013.keys():
    df_2013[i] = df_2013[i][df_2013[i].index.notna()]
    
df_2014 = pd.read_excel("data/datasets/2014.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2014.keys():
    df_2014[i] = df_2014[i][df_2014[i].index.notna()]
    
df_2015 = pd.read_excel("data/datasets/2015.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2015.keys():
    df_2015[i] = df_2015[i][df_2015[i].index.notna()]
    
df_2016 = pd.read_excel("data/datasets/2016.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2016.keys():
    df_2016[i] = df_2016[i][df_2016[i].index.notna()]
    
df_2017 = pd.read_excel("data/datasets/2017.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2017.keys():
    df_2017[i] = df_2017[i][df_2017[i].index.notna()]
    
df_2018 = pd.read_excel("data/datasets/2018.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2018.keys():
    df_2018[i] = df_2018[i][df_2018[i].index.notna()]
    
df_2019 = pd.read_excel("data/datasets/2019.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2019.keys():
    df_2019[i] = df_2019[i][df_2019[i].index.notna()]
    
df_2020 = pd.read_excel("data/datasets/2020.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2020.keys():
    df_2020[i] = df_2020[i][df_2020[i].index.notna()]
    
df_2021 = pd.read_excel("data/datasets/2021.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2021.keys():
    df_2021[i] = df_2021[i][df_2021[i].index.notna()]
    
df_2022 = pd.read_excel("data/datasets/2022.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2022.keys():
    df_2022[i] = df_2022[i][df_2022[i].index.notna()]
    
df_2023 = pd.read_excel("data/datasets/2023.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_2023.keys():
    df_2023[i] = df_2023[i][df_2023[i].index.notna()]

df_rtn = pd.read_excel("data/datasets/historical_return.xlsx",  \
                   index_col=0, header= 0, sheet_name=None)
for i in df_rtn.keys():
    df_rtn[i] = df_rtn[i][df_rtn[i].index.notna()]

df_benchmark = pd.read_excel("data/datasets/benchmark.xlsx", index_col=0).squeeze().reset_index(drop=True)

df_five_star = pd.read_excel("data/datasets/five_star_Morn.xlsx", index_col=0).squeeze().reset_index(drop=True)

df_four_star = pd.read_excel("data/datasets/four_star_Morn.xlsx", index_col=0).squeeze().reset_index(drop=True)


# Importing External libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_performances(performance1: pd.Series(), performance2: pd.Series(), names: list()):
    
    rtn1 = np.log(performance1 / performance1.shift(1)).fillna(0).reset_index(drop=True)
    rtn2 = np.log(performance2 / performance2.shift(1)).fillna(0).reset_index(drop=True)
    
    port1 = pd.Series([100])
    port2 = pd.Series([100])

    for i in range(len(rtn1)):
        value1 = (port1.iloc[-1]*(1+rtn1[i]))
        value2 = (port2.iloc[-1])*(1+rtn2[i])
        
        port1 = pd.concat([port1, pd.Series(value1)], axis=0, ignore_index=True).reset_index(drop=True)
        port2 = pd.concat([port2, pd.Series(value2)], axis=0, ignore_index=True).reset_index(drop=True)
            
    plt.plot(port1, label= names[0], color='blue')
    plt.plot(port2, label= names[1], color='orange')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.legend()
    
    return plt.show()


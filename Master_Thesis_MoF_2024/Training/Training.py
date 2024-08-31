# Importing External Libraries
import pandas as pd

# Importing Internal Libraries
from Training.methods.train_logistic_regression \
    import train_logistic_regression
from Training.methods.train_knn import train_knn
from Training.methods.train_random_forest import train_random_forest
from Training.methods.train_svm import train_svm
from Training.methods.train_gaussian_nb_c import train_gaussian_nb_c
from Training.methods.train_NN import train_NN

"""
Setting up class to prepare data for analysis
"""
class Training:
    """
    Class dedicated to training ML algorithms
    ***important*** All the methods include during the training phase, 
    the tunining of the hypeparameters, thus, the returned model is always
    the "best" one available.
    
    ***Methods***
    1. train_logistic_regression: Trains a Logistic Regression model
    
    2. train_knn: Trains a KNN model
    
    3. train_random_forest: Trains a Random forest ensemble method
    
    4. train_svm: trains a Support Vector CLassifier
    
    5. train_gaussian_nb_c: trains a GaussianNB classifier
    
    6. train_NN: Trains an MLP classifier
    """
    
    def train_logistic_regression(X: pd.DataFrame(), y: pd.Series()):
        
        return train_logistic_regression(X, y)
    
    def train_knn(X: pd.DataFrame(), y: pd.Series()):
        
        return train_knn(X, y)
    
    def train_random_forest(X: pd.DataFrame(), y: pd.Series()):
        
        return train_random_forest(X, y)
    
    def train_svm(X: pd.DataFrame(), y: pd.Series()):
        
        return train_svm(X, y)
    
    def train_gaussian_nb_c(X: pd.DataFrame(), y: pd.Series()):
        
        return train_gaussian_nb_c(X, y)
    
    def train_NN(X: pd.DataFrame(), y: pd.Series()):
        
        return  train_NN(X, y)   
    
    

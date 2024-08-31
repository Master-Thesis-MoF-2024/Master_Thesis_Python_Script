# Importing external libraries
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X, y, random_state = 42):
    
    # Create a pipeline with MinMaxScaler (Normalization) and KNeighborsClassifier
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  
        ('classifier', KNeighborsClassifier()) 
    ])


    # Define the hyperparameter grid for KNN
    param_grid = {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],  
        'classifier__weights': ['uniform', 'distance'], 
        'classifier__p': [1, 2]  
    }
    
    
    # Initialize GridSearchCV with the pipeline
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    
    # Fit GridSearchCV
    grid_search.fit(X, y)
    
    # Best hyperparameters
    grid_search.best_params_
    
    # Best model
    tuned_model = grid_search.best_estimator_
    
    return tuned_model
    
    
    
# Importing external libraries
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def train_svm(X, y, random_state = 32):
    # Create a pipeline with StandardScaler and SVC (Support Vector Classifier)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('classifier', SVC())  
    ])
    
    # Define the hyperparameter grid for SVM
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],  
        'classifier__kernel': ['linear', 'rbf', 'poly'],  
        'classifier__gamma': ['scale', 'auto'],  
        'classifier__degree': [3, 4, 5]  
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

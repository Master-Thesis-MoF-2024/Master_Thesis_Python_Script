# Importing external libraries
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X, y, random_state = 42):
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=random_state))  
    ])
    
    # Define the hyperparameter grid for Random Forest
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],  
        'classifier__max_depth': [None, 10, 20, 30],  
        'classifier__min_samples_split': [2, 5, 10], 
        'classifier__min_samples_leaf': [1, 2, 4],  
        'classifier__max_features': ['auto', 'sqrt', 'log2']  
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
        
    
    
    
    
    
    
    
    
    
    
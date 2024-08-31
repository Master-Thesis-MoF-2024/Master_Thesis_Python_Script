# Importing External Libraries
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X, y, random_state = 52):
    
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('classifier', LogisticRegression(random_state=random_state))  
    ])
    
    param_grid = {
    'classifier__penalty': ['l1', 'l2'], 
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'saga']  
    }
    
    # Initialize GridSearchCV with the pipeline
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    
    # Fit GridSearchCV
    grid_search.fit(X, y)
    

    grid_search.best_params_
    
    # Best model
    tuned_model = grid_search.best_estimator_
    
    return tuned_model






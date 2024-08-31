# Importing external libraries
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

def train_gaussian_nb_c(X, y, random_state = 42):
    # Step 3: Create Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])
    
    # Step 4: Define Hyperparameter Grid
    param_grid = {
        'classifier__var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05]
    }
    
    # Step 5: Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X, y)
    
    # Step 6: Evaluate Best Model
    grid_search.best_params_
    tuned_model = grid_search.best_estimator_
    
    return tuned_model
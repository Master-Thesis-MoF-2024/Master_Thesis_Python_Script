# Importing external libraries
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

def train_NN(X, y, random_state = 42):
    # Create a pipeline with StandardScaler and MLPClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),           
        ('classifier', MLPClassifier(random_state=42))  
    ])
    
   
    # Define the hyperparameter grid for MLPClassifier
    param_grid = {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  
        'classifier__activation': ['tanh', 'relu'],  
        'classifier__solver': ['adam', 'sgd'],  
        'classifier__alpha': [0.0001, 0.001, 0.01],  
        'classifier__learning_rate': ['constant', 'adaptive'],  
        'classifier__max_iter': [200, 400]  
    }
    
    # Initialize GridSearchCV with the pipeline and parameter grid
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,                        
        n_jobs=-1,                   
        verbose=2                    
    )
    
    # Fit GridSearchCV to the training data
    grid_search.fit(X, y)
    
    # Display the best hyperparameters
    grid_search.best_params_
    
    # Retrieve the best model
    tuned_model = grid_search.best_estimator_

    return tuned_model




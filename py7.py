import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV

# Load train and test data
path = "d:/study_data/_data/gas/"
path_save = "d:/study_data/_save/gas/"
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

train_data = train_data.drop(['out_pressure'],axis=1)
test_data = test_data.drop(['out_pressure'],axis=1)

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)

train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

# Train isolation forest model on train data
model = IsolationForest(random_state=3245, bootstrap=False)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_samples': [500, 1000, 2000],
    'max_features': [3, 5, 7]
}

# Perform grid search cross-validation to find best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(train_data)

# Print best hyperparameters and their corresponding scores
print("Best hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Use best hyperparameters to fit model on entire training set
best_model = grid_search.best_estimator_
best_model.fit(train_data)

# Predict anomalies in test data
predictions = best_model.predict(test_data)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})
submission.to_csv(path_save+'sjh5.csv', index=False)
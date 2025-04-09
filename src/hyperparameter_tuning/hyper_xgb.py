import os
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Function to set the seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)

set_seed(42)

# Load your dataset
# Assumes the features and labels are stored in .npy files
x = np.load("./../../data/features/x_1_3.npy")
y = np.load("./../../data/features/y_1_3.npy")
print("Data Shape:", x.shape)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training Data Shape:", x_train.shape)
print("Test Data Shape:", x_test.shape)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Flatten the data: adjust according to your actual data dimensions
n_samples_train = x_train.shape[0]
x_train_flat = x_train.reshape(n_samples_train, -1)

n_samples_test = x_test.shape[0]
x_test_flat = x_test.reshape(n_samples_test, -1)

# Define the directory and file path where the study results will be saved
study_dir = "/data/study_xgb"
os.makedirs(study_dir, exist_ok=True)
study_file = os.path.join(study_dir, "study_results_xgb.csv")

# Define the objective function for Optuna
def objective(trial):
    # Define the search space according to the given parameter grid
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
        "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10, 12]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1, 0.2, 0.3]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
        "gamma": trial.suggest_categorical("gamma", [0, 0.1, 0.3, 0.5]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.01, 0.1, 1]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [0.1, 1, 5, 10]),
        "min_child_weight": trial.suggest_categorical("min_child_weight", [1, 3, 5, 10]),
        # Fixed parameters
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42
    }
    
    # Instantiate the XGBoost classifier with the hyperparameters
    model = XGBClassifier(**params)
    
    # Evaluate with 3-fold cross-validation on the training set
    cv_scores = cross_val_score(model, x_train_flat, y_train_encoded, cv=3, scoring="accuracy")
    return np.mean(cv_scores)

# Create and run the Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

# Save the trial results to a CSV file
df = study.trials_dataframe()
df.to_csv(study_file, index=False)
print(f"Study results saved to {study_file}")

# Retrieve the best hyperparameters
best_params = study.best_params
print("Best Parameters:", best_params)

# Train the final model with the best parameters on the full training data
best_xgb = XGBClassifier(**best_params,
                         use_label_encoder=False,
                         eval_metric="mlogloss",
                         random_state=42)
best_xgb.fit(x_train_flat, y_train_encoded)

# Make predictions on the test set and evaluate performance
y_pred = best_xgb.predict(x_test_flat)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))
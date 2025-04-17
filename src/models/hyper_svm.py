import os
import numpy as np
import pandas as pd
import optuna
from sklearn.svm import SVC
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
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
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
study_dir = "/data/study_svm"
os.makedirs(study_dir, exist_ok=True)
study_file = os.path.join(study_dir, "study_results_svm.csv")


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for SVC
    svc_params = {
        "C": trial.suggest_loguniform("C", 1e-3, 1e3),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        "gamma": trial.suggest_loguniform("gamma", 1e-4, 1e1)
    }

    # If the selected kernel is polynomial, add the degree hyperparameter
    if svc_params["kernel"] == "poly":
        svc_params["degree"] = trial.suggest_int("degree", 2, 5)

    # Instantiate the SVM classifier with the suggested hyperparameters
    model = SVC(**svc_params, random_state=42)

    # Evaluate with 3-fold cross-validation on the training set
    cv_scores = cross_val_score(
        model, x_train_flat, y_train_encoded, cv=3, scoring="accuracy"
    )
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
best_svm = SVC(**best_params, random_state=42)
best_svm.fit(x_train_flat, y_train_encoded)

# Make predictions on the test set and evaluate performance
y_pred = best_svm.predict(x_test_flat)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:\n", classification_report(y_test_encoded, y_pred))

import numpy as np
# import pandas as pd
import librosa
# import tensorflow as tf
# from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import torch
# import torchaudio
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)

    # Ensures deterministic behavior (optional, can slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

x = np.load("./../../data/features/x_3_3.npy")
y = np.load("./../../data/features/y_3_3.npy")
input_shape = (43, 39)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print("Training Data Shape:", x_train.shape)
print("Test Data Shape:", x_test.shape)

# Initialize encoder and fit on full set of labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Flatten the data: (n_samples, 32, 13) to (n_samples, 32*13)
n_samples_train = x_train.shape[0]
x_train_flat = x_train.reshape(n_samples_train, -1)

n_samples_test = x_test.shape[0]
x_test_flat = x_test.reshape(n_samples_test, -1)

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import optuna

# Define the path to save the study
study_dir = "/data/study_rf"
os.makedirs(study_dir, exist_ok=True)
study_file = os.path.join(study_dir, "study_results_rf.csv")

# Flatten the data: (n_samples, 32, 13) to (n_samples, 32*13)
n_samples_train = x_train.shape[0]
x_train_flat = x_train.reshape(n_samples_train, -1)

n_samples_test = x_test.shape[0]
x_test_flat = x_test.reshape(n_samples_test, -1)


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    max_depth = trial.suggest_int("max_depth", 10, 50, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 10, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    bootstrap = trial.suggest_categorical("bootstrap", [False])

    # Initialize RandomForestClassifier
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1
    )

    # Evaluate using cross-validation
    cv_scores = cross_val_score(
        rf_clf, x_train_flat, y_train, cv=5, scoring="accuracy"
    )
    return np.mean(cv_scores)


# Create an Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25, n_jobs=25)

# Save study results to a CSV file
df = study.trials_dataframe()
df.to_csv(study_file, index=False)
print(f"Study results saved to {study_file}")

# Retrieve the best parameters
best_params = study.best_params
print("Best Parameters:", best_params)

# Train the best model
best_rf_clf = RandomForestClassifier(
    **best_params, random_state=42, n_jobs=-1
)
best_rf_clf.fit(x_train_flat, y_train)

# Predict and evaluate on the test set
y_pred = best_rf_clf.predict(x_test_flat)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", report)

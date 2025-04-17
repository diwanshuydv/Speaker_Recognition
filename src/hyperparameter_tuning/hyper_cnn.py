import numpy as np
import librosa
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

x = np.load("./../../data/features/x_3_3.npy")
y = np.load("./../../data/features/y_3_3.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
print("Training Data Shape:", x_train.shape)
print("Test Data Shape:", x_test.shape)
input_shape = x_train.shape[1:]

class SpeakerCNN(nn.Module):
    def __init__(self, input_shape, no_speakers, dropout_rate=0.5):
        super(SpeakerCNN, self).__init__()
        
        self.time_frames, self.mfcc_features = input_shape

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual Block 1
        self.res1_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.res1_bn1 = nn.BatchNorm2d(64)
        self.res1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res1_bn2 = nn.BatchNorm2d(64)
        self.res1_shortcut = nn.Conv2d(32, 64, kernel_size=1)

        # Residual Block 2
        self.res2_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res2_bn1 = nn.BatchNorm2d(128)
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2_bn2 = nn.BatchNorm2d(128)
        self.res2_shortcut = nn.Conv2d(64, 128, kernel_size=1)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, no_speakers)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension

        # Convolutional Block 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Block 1
        shortcut = self.res1_shortcut(x)
        x = F.relu(self.res1_bn1(self.res1_conv1(x)))
        x = self.res1_bn2(self.res1_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Residual Block 2
        shortcut = self.res2_shortcut(x)
        x = F.relu(self.res2_bn1(self.res2_conv1(x)))
        x = self.res2_bn2(self.res2_conv2(x))
        x += shortcut
        x = F.relu(x)

        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

batch_size = 128
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

import optuna
from sklearn.metrics import accuracy_score

# Define the objective function
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.2, 0.8)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    # Model initialization with suggested dropout rate
    model = SpeakerCNN(input_shape=input_shape, no_speakers=51, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Define optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Create DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Validation loop
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, axis=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(y_batch.cpu().numpy())
    
    # Calculate validation accuracy
    accuracy = accuracy_score(val_targets, val_preds)
    return accuracy

# Create an Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=25)

# Save study results to a CSV file
study_file = "optuna_study_results_cnn.csv"
df = study.trials_dataframe()
df.to_csv(study_file, index=False)
print(f"Study results saved to {study_file}")

# Retrieve the best parameters
best_params = study.best_params
print("Best Parameters:", best_params)

# Train the final model with the best parameters
final_model = SpeakerCNN(input_shape=input_shape, no_speakers=51, dropout_rate=best_params["dropout_rate"])
final_model = final_model.to(device)
optimizer = getattr(optim, best_params["optimizer"])(final_model.parameters(), lr=best_params["learning_rate"])
criterion = nn.CrossEntropyLoss()
batch_size = best_params["batch_size"]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the final model
num_epochs = 30
for epoch in range(num_epochs):
    final_model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = final_model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Evaluate on the test set
final_model.eval()
test_preds = []
test_targets = []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = final_model(x_batch)
        preds = torch.argmax(outputs, axis=1).cpu().numpy()
        test_preds.extend(preds)
        test_targets.extend(y_batch.cpu().numpy())

test_accuracy = accuracy_score(test_targets, test_preds)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))

# Save the final model
torch.save(final_model.state_dict(), "final_model_3_3.pth")
# Save the model architecture
torch.save(final_model, "final_model_architecture_3_3.pth")

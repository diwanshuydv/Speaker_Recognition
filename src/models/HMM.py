import numpy as np
from hmmlearn import hmm

# Load data
x = np.load("./../../data/features/x_raw.npy")  # Observations
y = np.load("./../../data/features/y_raw.npy")  # Labels (hidden states)

# Check the data shapes
print(f"x shape: {x.shape}, y shape: {y.shape}")

# Reshape the data
n_sequences, n_timesteps, n_features = x.shape
x_flattened = x.reshape(-1, n_features)  # Combine all timesteps

# Generate sequence lengths
sequence_lengths = [n_timesteps] * n_sequences

# Determine number of hidden states
num_states = len(np.unique(y)) if y is not None else 3

# Initialize HMM
model = hmm.GaussianHMM(
    n_components=num_states,
    covariance_type="diag",
    n_iter=100,
    random_state=42
)

# Train the model
print("Training the HMM...")
model.fit(x_flattened, lengths=sequence_lengths)

# Predict the hidden states
predicted_states = model.predict(x_flattened, lengths=sequence_lengths)
print(f"Predicted states shape: {predicted_states.shape}")

# Evaluate accuracy if ground truth is available
if y is not None:
    predicted_sequences = np.array_split(predicted_states, n_sequences)
    accuracy_per_sequence = [
        np.mean(pred == true)
        for pred, true in zip(predicted_sequences, y)
    ]
    overall_accuracy = np.mean(accuracy_per_sequence)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

# Generate a sample sequence
print("Generating a sample sequence...")
sample, _ = model.sample(10)
print(f"Sampled sequence shape: {sample.shape}")

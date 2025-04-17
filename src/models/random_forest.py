import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Flatten the data: (n_samples, 32, 13) to (n_samples, 32*13)
n_samples_train = x_train.shape[0]
x_train_flat = x_train.reshape(n_samples_train, -1)

n_samples_test = x_test.shape[0]
x_test_flat = x_test.reshape(n_samples_test, -1)

# Initialize the RandomForestClassifier
# You can tune n_estimators and other hyperparameters as needed.
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
rf_clf.fit(x_train_flat, y_train)

# Predict on the test set
y_pred = rf_clf.predict(x_test_flat)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", report)

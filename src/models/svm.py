import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Assuming x_train, x_test, y_train, y_test are already defined
# Flatten the features: from (n_samples, 32, 13) to (n_samples, 32*13)
n_samples_train = x_train.shape[0]
x_train_flat = x_train.reshape(n_samples_train, -1)

n_samples_test = x_test.shape[0]
x_test_flat = x_test.reshape(n_samples_test, -1)

# Create an SVM classifier; you can adjust the kernel and hyperparameters as needed
svm_clf = SVC(kernel='rbf', C=1.0, random_state=42)

# Train the classifier
svm_clf.fit(x_train_flat, y_train)

# Predict on the test set
y_pred = svm_clf.predict(x_test_flat)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Test Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", report)
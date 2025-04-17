from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Initialize the XGBoost classifier
xgb_clf = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)

# Train the classifier
xgb_clf.fit(x_train_flat, y_train)

# Predict on the test set
y_pred_xgb = xgb_clf.predict(x_test_flat)

# Evaluate the classifier
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Test Accuracy: {:.2f}%".format(xgb_accuracy * 100))

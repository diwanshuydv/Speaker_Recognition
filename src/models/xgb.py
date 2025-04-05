from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
xgb_clf.fit(x_train_flat, y_train)

y_pred_xgb = xgb_clf.predict(x_test_flat)
print("XGBoost Test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_xgb) * 100))
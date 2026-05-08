import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('heart.csv')
# Use only specified features: age, sex, trestbps, chol, fbs
features = ['age', 'sex', 'trestbps', 'chol', 'fbs']
X = data[features]
y = data['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Feature importance
feature_importance = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)
print('Feature Importance:')
print(importance_df)

# Save model
joblib.dump(model, 'heart_model.pkl')
print('Model saved as heart_model.pkl')

# Save evaluation metrics for later use
with open('model_metrics.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}\n')
    f.write(classification_report(y_test, y_pred))
    f.write('\nConfusion Matrix:\n')
    f.write(str(cm))
    f.write('\nFeature Importance:\n')
    f.write(str(importance_df))
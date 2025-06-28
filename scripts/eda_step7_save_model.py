import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load cleaned and encoded data
df = pd.read_csv('data/heart_cleaned.csv')
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang'], drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Train logistic regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

# Create directory if not exists
import os
os.makedirs("outputs/models", exist_ok=True)

# Save model and feature columns
joblib.dump(logreg_model, 'outputs/models/logreg_heart_model.pkl')
joblib.dump(X.columns.tolist(), 'outputs/models/logreg_features.pkl')

print("Logistic Regression model and feature list saved.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/heart_cleaned.csv')
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang'], drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Fit model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Get feature importance (coefficients)
coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=coefficients, y=coefficients.index, palette='coolwarm')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig('outputs/plots/logreg_feature_importance.png')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv('data/heart_cleaned.csv')

# 1. One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang'], drop_first=True)

# 2. Compute correlation matrix
corr = df_encoded.corr()

# 3. Sort features by correlation with target (absolute value for relevance)
corr_target_sorted = corr['target'].abs().sort_values(ascending=False)
sorted_features = corr_target_sorted.index.tolist()

# 4. Reorder the correlation matrix
corr_sorted = corr.loc[sorted_features, sorted_features]

# 5. Plot enhanced correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_sorted, cmap='coolwarm', annot=True, fmt=".2f", square=True, cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap (Sorted by Relevance to Target)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/plots/correlation_heatmap.png')
plt.close()

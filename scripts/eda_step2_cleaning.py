import pandas as pd

# Load data
df = pd.read_csv('data/heart.csv')

# 1. Drop columns that aren't useful
df.drop(columns=['id', 'dataset'], inplace=True)

# 2. Convert 'num' to binary target: 0 = no disease, 1 = disease
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns='num', inplace=True)

# 3. Check missing values %
missing_percent = df.isnull().mean().sort_values(ascending=False)
print("\nMissing Values (%):")
print((missing_percent * 100).round(2))

# 4. Drop columns with >40% missing data
df.drop(columns=['thal', 'ca', 'slope'], inplace=True)

# 5. Drop rows with any remaining nulls
df.dropna(inplace=True)

# 6. Print cleaned shape and sample
print("\nCleaned Data Shape:", df.shape)
print("\nSample after cleaning:")
print(df.head())

# 7. Save cleaned data (optional)
df.to_csv('data/heart_cleaned.csv', index=False)

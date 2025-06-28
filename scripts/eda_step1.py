import pandas as pd

# Load the dataset
df = pd.read_csv('data/heart.csv')  # make sure the CSV is in the 'data/' folder

# Basic inspection
print("Dataset Shape:", df.shape)
print("\nColumn Info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())
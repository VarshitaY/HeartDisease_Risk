import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv('data/heart_cleaned.csv')

# Set universal plot style
sns.set(style="whitegrid", palette="Set2")

# Create output directory if not already there
import os
os.makedirs('outputs/plots', exist_ok=True)

# 1. Heart Disease by Sex (normalized)
plt.figure(figsize=(6, 4))
sex_counts = df.groupby(['sex', 'target']).size().reset_index(name='count')
sns.barplot(data=sex_counts, x='sex', y='count', hue='target')
plt.title('Heart Disease by Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/plots/heart_disease_by_sex.png')
plt.close()

# 2. Age Distribution by Heart Disease Status
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', hue='target', kde=True, bins=20, element='step', stat='density', common_norm=False)
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('outputs/plots/age_distribution.png')
plt.close()

# 3. Max Heart Rate (thalch) vs Heart Disease
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='target', y='thalch')
plt.title('Max Heart Rate vs Heart Disease')
plt.xlabel('Target (0 = No Disease, 1 = Disease)')
plt.ylabel('Maximum Heart Rate Achieved')
plt.tight_layout()
plt.savefig('outputs/plots/thalch_vs_target.png')
plt.close()

# 4. ST Depression (oldpeak) vs Heart Disease
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='target', y='oldpeak')
plt.title('ST Depression (oldpeak) vs Heart Disease')
plt.xlabel('Target (0 = No Disease, 1 = Disease)')
plt.ylabel('Oldpeak (ST Depression)')
plt.tight_layout()
plt.savefig('outputs/plots/oldpeak_vs_target.png')
plt.close()

# 5. Exercise-induced Angina
plt.figure(figsize=(6, 4))
angina_counts = df.groupby(['exang', 'target']).size().reset_index(name='count')
sns.barplot(data=angina_counts, x='exang', y='count', hue='target')
plt.title('Heart Disease by Exercise-induced Angina')
plt.xlabel('Exercise-induced Angina (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/plots/exang_vs_target.png')
plt.close()

# 6. Chest Pain Type vs Heart Disease
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='cp', hue='target')
plt.title('Chest Pain Type vs Heart Disease')
plt.xlabel('Chest Pain Type (0–3)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2, 3], labels=[
    'Typical Angina', 
    'Atypical Angina', 
    'Non-anginal Pain', 
    'Asymptomatic'
])
plt.tight_layout()
plt.savefig('outputs/plots/chest_pain_vs_target.png')
plt.close()

print("✅ All plots successfully updated and saved to outputs/plots/")

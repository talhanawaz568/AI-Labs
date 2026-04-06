import pandas as pd

df = pd.read_csv('sample_data.csv')

# Perform one-hot encoding on the 'department' column
df_encoded = pd.get_dummies(df, columns=['department'])

print("--- Data after One-Hot Encoding ---")
# This will create columns like department_HR, department_IT, etc.
print(df_encoded.head())

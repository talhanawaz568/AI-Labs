import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('sample_data.csv')

# We'll use 'experience_years' for normalization
scaler = MinMaxScaler()

# Fit and transform
# reshape is handled automatically by passing a DataFrame slice [[ ]]
df['normalized_experience'] = scaler.fit_transform(df[['experience_years']])

print("--- Original vs Normalized Columns ---")
print(df[['experience_years', 'normalized_experience']].head())

import pandas as pd

# 1. Load the dataset
df = pd.read_csv('sample_data.csv')

# 2. Display missing values
print("--- Missing Values Count ---")
print(df.isnull().sum())

# 3. Drop Missing Values
df_dropped = df.dropna()
print("\n--- Data after dropping rows with any NULLs ---")
print(df_dropped)

# 4. Fill Missing Values
# Note: We only fill numeric columns with mean to avoid errors on 'name'
numeric_cols = df.select_dtypes(include=['number']).columns
df_filled = df.copy()
df_filled[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

print("\n--- Data after filling NULLs with Column Mean ---")
print(df_filled)

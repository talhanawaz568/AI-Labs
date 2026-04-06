import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())


print(df.describe())
print(df.isnull().sum())
df_cleaned = df.dropna()
print(df_cleaned.isnull().sum())

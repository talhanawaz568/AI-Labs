import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # For Ubuntu headless environment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Load and Explore the Dataset ---
print("Step 1: Fetching California Housing Data...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target # This is our 'Target' (Price in $100k units)

print(f"Dataset Shape: {df.shape}")
print(df.head())

# --- 2. Data Cleaning ---
# Check for missing values (The built-in dataset is clean, but we check anyway)
print("\nStep 2: Cleaning Data...")
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)
    print("✓ Missing values filled with mean.")
else:
    print("✓ No missing values found.")

# --- 3. Feature Engineering ---
print("Step 3: Scaling Features...")
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Model Selection & Training ---
print("Step 4: Training Random Forest Regressor...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Using 100 trees for our ensemble
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. Evaluate and Present Results ---
print("\nStep 5: Evaluating Model...")
predictions = model.predict(X_test)

# RMSE tells us the average error
rmse = np.sqrt(mean_squared_error(y_test, predictions))
# R2 Score tells us how well the model 'fits' the data (1.0 is perfect)
r2 = r2_score(y_test, predictions)

print("-" * 40)
print(f"FINAL REPORT")
print("-" * 40)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-Squared (Accuracy Metric):    {r2:.4f}")
print("-" * 40)

# Visualization: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs Predicted House Values')
plt.xlabel('Actual Value ($100k)')
plt.ylabel('Predicted Value ($100k)')
plt.savefig('final_project_results.png')
print("✓ Performance chart saved as 'final_project_results.png'")

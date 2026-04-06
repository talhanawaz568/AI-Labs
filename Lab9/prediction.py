import pandas as pd
import numpy as np
import matplotlib
# Force Matplotlib to use 'Agg' backend for headless Ubuntu
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Prepare the Dataset
data = {
    'Size': [1500, 1600, 1700, 1800, 1900, 2000, 2100],
    'Price': [250000, 270000, 290000, 310000, 330000, 350000, 370000]
}
df = pd.DataFrame(data)

print("--- House Price Dataset ---")
print(df)
print("-" * 25)

# 2. Define Features (X) and Target (y)
X = df[['Size']] # Features must be 2D (DataFrame)
y = df['Price'] # Target can be 1D (Series)

# 3. Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

coefficient = model.coef_[0]
intercept = model.intercept_

print(f"Coefficient (Slope): {coefficient}")
print(f"Intercept: {intercept}")

# 4. Predict a new value (Example: 2500 sq ft house)
new_size = np.array([[2500]])
predicted_price = model.predict(new_size)
print(f"\nPredicted Price for 2500 sq ft: ${predicted_price[0]:,.2f}")

# 5. Visualization
plt.figure(figsize=(10, 6))

# Plot the actual data points
plt.scatter(X, y, color='blue', label='Actual Data Points')

# Plot the regression line (Line of best fit)
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

plt.xlabel('Size (sq ft)')
plt.ylabel('Price (USD)')
plt.title('Simple Linear Regression: House Size vs Price')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 6. Save the output
output_file = 'house_price_regression.png'
plt.savefig(output_file)
print(f"\n✓ Task Complete. Plot saved as '{output_file}'")

import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for headless environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import os

# --- Task 1: Load/Generate Dataset ---
file_path = 'daily_temperatures.csv'

if not os.path.exists(file_path):
    print("Generating synthetic temperature data...")
    dates = pd.date_range(start='2022-01-01', end='2023-12-31')
    # Generate a seasonal pattern (sine wave) + noise
    temp = 20 + 10 * np.sin(np.arange(len(dates)) * (2 * np.pi / 365)) + np.random.normal(0, 2, len(dates))
    data = pd.DataFrame({'Date': dates, 'Temperature': temp})
    data.to_csv(file_path, index=False)

# Load data
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
print("First 5 rows of data:\n", data.head())

# --- Task 2: Split into Train/Test ---
# We split chronologically, not randomly!
train_data = data[:'2022-12-31']
test_data = data['2023-01-01':]

print(f"\nTraining data: {len(train_data)} points")
print(f"Testing data:  {len(test_data)} points")

# --- Task 3: Fit ARIMA Model ---
print("\nFitting ARIMA(5,1,0) model...")
# order=(p,d,q) 
# p: Lag observations, d: Differencing, q: Moving average window
model = ARIMA(train_data['Temperature'], order=(5,1,0))
model_fit = model.fit()

# Forecast for the duration of the test set
forecast = model_fit.forecast(steps=len(test_data))

# --- Evaluation & Visualization ---
mse = mean_squared_error(test_data['Temperature'], forecast)
print(f"Mean Squared Error: {mse:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Temperature'], label='Training Data')
plt.plot(test_data.index, test_data['Temperature'], label='Actual (Test)')
plt.plot(test_data.index, forecast, label='ARIMA Forecast', color='red', linestyle='--')
plt.title('Daily Temperature Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('forecast_results.png')
print("\n✓ Forecast plot saved as 'forecast_results.png'")

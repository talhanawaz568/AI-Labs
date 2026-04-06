import numpy as np
import pandas as pd
import matplotlib
# Force Matplotlib to use 'Agg' backend for headless Ubuntu
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Prepare Sample Data
# X = Study Hours, y = Pass (1) or Fail (0)
data = {
    'StudyHours': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    'Pass': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[['StudyHours']]
y = df['Pass']

# 2. Split and Train the Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 3. Visualization
plt.figure(figsize=(10, 6))

# Plot the actual test data points
plt.scatter(X_test, y_test, color='red', label='Actual Data (0=Fail, 1=Pass)')

# Plot the Linear Regression Line
# This shows the linear trend of passing as study hours increase
plt.plot(X, model.predict(X), color='blue', linewidth=2, label='Linear Regression Line')

# Add a threshold line at 0.5 (often used as the pass/fail cutoff)
plt.axhline(y=0.5, color='green', linestyle='--', label='50% Threshold')

plt.xlabel('Study Hours')
plt.ylabel('Predicted Pass Score (0 to 1)')
plt.title('Simple Linear Regression: Predicting Pass/Fail')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Save the output
output_file = 'linear_pass_fail.png'
plt.savefig(output_file)
print(f"✓ Task Complete. Plot saved as '{output_file}'")

# 5. Prediction Example
test_hours = np.array([[3.5]])
prediction = model.predict(test_hours)[0]
print(f"Predicted score for 3.5 hours: {prediction:.2f}")
print(f"Outcome: {'Pass' if prediction >= 0.5 else 'Fail'}")

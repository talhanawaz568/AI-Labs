import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Task 2: Generate and Prepare Dataset ---
print("Task 2: Generating synthetic dataset...")
# We use a specific random seed so your results are reproducible
np.random.seed(42)

data = {
    'Feature1': np.random.randint(0, 100, 100),    # Small range
    'Feature2': np.random.randint(1000, 5000, 100), # Much larger range
    'Target': np.random.choice([0, 1], 100)
}
df = pd.DataFrame(data)

X = df[['Feature1', 'Feature2']]
y = df['Target']

# --- Task 3.1 & 3.2: Train BEFORE Scaling ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_raw = LogisticRegression()
model_raw.fit(X_train, y_train)
y_pred = model_raw.predict(X_test)
original_accuracy = accuracy_score(y_test, y_pred)

# --- Task 2.3: Standardize features ---
print("\nTask 2.3: Applying StandardScaler...")
scaler = StandardScaler()
# Note: In production, you fit on Train and transform Test, but for this lab 
# we follow the instruction to scale the whole set X first.
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# --- Task 3.3 & 3.4: Train AFTER Scaling ---
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

model_scaled = LogisticRegression()
model_scaled.fit(X_train_s, y_train_s)
y_pred_s = model_scaled.predict(X_test_s)
scaled_accuracy = accuracy_score(y_test_s, y_pred_s)

# --- Results Comparison ---
print("\n" + "="*40)
print(f"{'METRIC':<25} | {'VALUE':<10}")
print("-" * 40)
print(f"{'Accuracy (Unscaled)':<25} | {original_accuracy:.4f}")
print(f"{'Accuracy (Standardized)':<25} | {scaled_accuracy:.4f}")
print("="*40)

print("\nTask 3.5: Findings")
print("Standardization transforms data to have a Mean = 0 and Std Dev = 1.")
print("This ensures that 'Feature2' doesn't dominate the model simply because")
print("its raw numbers are larger.")

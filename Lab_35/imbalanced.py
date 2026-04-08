import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless Ubuntu environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# --- Task 1: Load/Generate an Imbalanced Dataset ---
print("Task 1: Generating synthetic imbalanced data (99% vs 1%)...")
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.99, 0.01], 
                           n_informative=3, n_redundant=1, flip_y=0, 
                           n_features=5, n_clusters_per_class=1, n_samples=1000, random_state=42)

# Check the distribution
counts = np.bincount(y)
print(f"Original distribution: Class 0: {counts[0]}, Class 1: {counts[1]}")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Task 3.2: Performance BEFORE Balancing ---
print("\nTask 3.2: Evaluating Model on Imbalanced Data...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("-" * 30)
print("Report - Imbalanced Data:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- Task 2: Apply SMOTE (Oversampling) ---
print("\nTask 2: Applying SMOTE to balance the training set...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

res_counts = np.bincount(y_res)
print(f"Balanced distribution: Class 0: {res_counts[0]}, Class 1: {res_counts[1]}")

# --- Task 3.3: Performance AFTER Balancing ---
print("\nTask 3.3: Evaluating Model on Balanced Data...")
model_res = LogisticRegression(random_state=42)
model_res.fit(X_res, y_res)
y_pred_res = model_res.predict(X_test)
print("-" * 30)
print("Report - Balanced Data (SMOTE):")
print(classification_report(y_test, y_pred_res))

# --- Visualization (Optional) ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['Class 0', 'Class 1'], counts, color=['blue', 'red'])
plt.title("Before SMOTE")

plt.subplot(1, 2, 2)
plt.bar(['Class 0', 'Class 1'], res_counts, color=['blue', 'red'])
plt.title("After SMOTE")

plt.savefig('imbalance_comparison.png')
print("\n✓ Comparison plot saved as 'imbalance_comparison.png'")

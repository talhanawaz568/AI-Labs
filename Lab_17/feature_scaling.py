import pandas as pd
import numpy as np
import matplotlib
# Force Matplotlib to use 'Agg' backend for headless Ubuntu
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Section 1: Data Preparation ---
print("Task 1: Loading Iris dataset and splitting data...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Add a "Noise" feature to see if the model correctly identifies it as unimportant
X['random_noise'] = np.random.normal(0, 1, X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Section 2: Rank Features by Importance ---
print("Task 2: Training Random Forest to rank features...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting
plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='skyblue', edgecolor='navy')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✓ Feature importance plot saved as 'feature_importance.png'")

# --- Section 3: Removing Low-Importance Features ---
print("\nTask 3: Removing low-importance features (Threshold < 0.05)...")
threshold = 0.05
low_imp_cols = [X.columns[i] for i in range(X.shape[1]) if importances[i] < threshold]
print(f"Features to remove: {low_imp_cols}")

X_train_reduced = X_train.drop(columns=low_imp_cols)
X_test_reduced = X_test.drop(columns=low_imp_cols)

# --- Section 4: Performance Comparison ---
# 4.1 Original Model
y_pred_orig = model.predict(X_test)
acc_orig = accuracy_score(y_test, y_pred_orig)

# 4.2 Reduced Model
model_red = RandomForestClassifier(n_estimators=100, random_state=42)
model_red.fit(X_train_reduced, y_train)
y_pred_red = model_red.predict(X_test_reduced)
acc_red = accuracy_score(y_test, y_pred_red)

print("\n" + "="*45)
print(f"{'MODEL CONFIGURATION':<25} | {'ACCURACY':<10}")
print("-" * 45)
print(f"{'All Features (' + str(X.shape[1]) + ')':<25} | {acc_orig:.4f}")
print(f"{'Selected Features (' + str(X_train_reduced.shape[1]) + ')':<25} | {acc_red:.4f}")
print("="*45)

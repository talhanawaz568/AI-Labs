import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# --- Task 1: Load and Split the Dataset ---
print("Step 1: Preparing Iris Dataset...")
data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Task 2: Single Decision Tree (The Baseline) ---
print("\nTraining Single Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

# --- Task 1: Random Forest (Bagging) ---
print("Training Random Forest (Bagging)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# --- Task 3: Gradient Boosting (Boosting) ---
print("Training Gradient Boosting (Boosting)...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred_gb)

# --- Final Comparison ---
print("\n" + "="*45)
print(f"{'Model Type':<25} | {'Accuracy':<10}")
print("-" * 45)
print(f"{'Single Decision Tree':<25} | {dt_accuracy:.4f}")
print(f"{'Random Forest (Bagging)':<25} | {rf_accuracy:.4f}")
print(f"{'Gradient Boosting':<25} | {gb_accuracy:.4f}")
print("="*45)

# Task 2.3 & 3.3: Analysis Output
print("\nLab Analysis:")
print(f"1. Decision Tree vs Random Forest: The Forest is usually more stable.")
print(f"   By using 100 trees, we reduce the chance of the model being 'tricked' by noise.")
print(f"2. Boosting Performance: Gradient Boosting works to minimize error step-by-step.")
print(f"   In complex datasets, Boosting often becomes the 'Gold Standard' for accuracy.")

import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend to save plots as files in a headless Ubuntu environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import shap

# --- Task 2: Setup and Train Model ---
print("Task 2: Loading Iris dataset and training Random Forest...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✓ Model trained successfully.")

# --- Task 2.1: Interpret with SHAP ---
print("\nTask 2.1: Computing SHAP values (TreeExplainer)...")
# Create a SHAP explainer specifically for tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot 1: Summary Plot (Global Interpretability)
# This shows the importance of each feature across all test samples
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot - Global Importance")
plt.savefig('shap_summary_plot.png')
print("✓ Global summary plot saved as 'shap_summary_plot.png'")

# --- Task 3: Explanation for a Sample Prediction ---
print("\nTask 3: Explaining a single prediction (Local Interpretability)...")
sample_index = 0
# For multi-class (Iris has 3 classes), we look at SHAP values for class 0
# shap_values[class_index][sample_index]
sample_shap = shap_values[0][sample_index] 

# Plot 2: Bar Plot for the first sample
plt.clf() # Clear figure
plt.figure(figsize=(8, 4))
shap.bar_plot(sample_shap, feature_names=feature_names, show=False)
plt.title(f"Feature Contributions for Sample {sample_index}")
plt.savefig('shap_sample_explanation.png')

print(f"✓ Sample prediction explanation saved as 'shap_sample_explanation.png'")

# --- Conclusion Description ---
print("\n" + "="*40)
print("LAB ANALYSIS")
print("="*40)
print("- The Summary Plot shows which features have the most 'push' on the model.")
print("- Petal Length and Petal Width are typically the most influential features.")
print("- SHAP values prove that the model isn't just guessing; it's weighing")
print("  specific physical characteristics to make its choice.")
print("="*40)

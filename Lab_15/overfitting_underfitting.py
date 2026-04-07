import numpy as np
import matplotlib
# Force Matplotlib to use 'Agg' backend for headless environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os

# --- Task 1: Load and Split the Dataset ---
print("Task 1: Preparing data...")
iris = load_iris()
X, y = iris.data, iris.target

# Using a 30% split for validation to better observe performance gaps
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Task 2: Adjust Model Complexity (Max Depth) ---
depth_range = range(1, 11)
train_scores = []
val_scores = []

print("\nTask 2: Training models with increasing complexity...")
print(f"{'Depth':<10} | {'Train Acc':<12} | {'Val Acc':<12}")
print("-" * 40)

for depth in depth_range:
    # Initialize model with specific depth
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    t_score = model.score(X_train, y_train)
    v_score = model.score(X_val, y_val)
    
    train_scores.append(t_score)
    val_scores.append(v_score)
    
    print(f"{depth:<10} | {t_score:<12.4f} | {v_score:<12.4f}")

# --- Task 3: Visualize and Analyze ---
plt.figure(figsize=(10, 6))
plt.plot(depth_range, train_scores, label='Training Score (Accuracy)', marker='o', linewidth=2)
plt.plot(depth_range, val_scores, label='Validation Score (Generalization)', marker='s', linewidth=2)

plt.xlabel('Model Complexity (Max Depth)')
plt.ylabel('Accuracy Score')
plt.title('Detection of Overfitting and Underfitting')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Highlight the "Sweet Spot"
plt.annotate('Optimal Complexity', xy=(2, val_scores[1]), xytext=(4, 0.85),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Save the plot
output_file = 'fit_analysis_plot.png'
plt.savefig(output_file)
print(f"\n✓ Task Complete. Analysis plot saved as '{output_file}'")

print("\n--- Lab Report Summary ---")
print("1. Underfitting: Seen at Depth 1 (Low scores on both Train and Val).")
print("2. Overfitting: Occurs when Train Score stays at 1.000 but Val Score drops or plateaus.")
print("3. Generalization: The depth where Val Score is highest is your production-ready model.")

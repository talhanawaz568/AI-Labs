import matplotlib
# Force Matplotlib to use 'Agg' backend for headless Ubuntu CLI
matplotlib.use('Agg') 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# --- Task 1: Load and Split Dataset ---
print("Task 1: Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split complete. Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- Task 2: Train a Initial KNN Classifier ---
print("\nTask 2: Training initial KNN with k=3...")
knn_default = KNeighborsClassifier(n_neighbors=3)
knn_default.fit(X_train, y_train)
initial_pred = knn_default.predict(X_test)
print(f"Initial Accuracy (k=3): {accuracy_score(y_test, initial_pred):.4f}")

# --- Task 3: Test Different Values of k ---
def evaluate_k(k_values, xt, yt, xv, yv):
    accuracies = []
    for k in k_values:
        # Initialize and fit
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(xt, yt)
        # Predict and calculate accuracy
        preds = model.predict(xv)
        acc = accuracy_score(yv, preds)
        accuracies.append(acc)
        print(f"Testing k={k:2} | Accuracy: {acc:.4f}")
    return accuracies

print("\nTask 3: Experimenting with k values 1 through 15...")
k_range = range(1, 16)
accuracies = evaluate_k(k_range, X_train, y_train, X_test, y_test)

# --- Task 3.3: Plot the Results ---
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o', linestyle='dashed', color='blue', markerfacecolor='red')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Model Accuracy')
plt.title('KNN Accuracy Comparison for Different k Values')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
output_img = "knn_accuracy_plot.png"
plt.savefig(output_img)
plt.close()

print(f"\n--- Lab 11 Complete ---")
print(f"Visualization saved to: {os.getcwd()}/{output_img}")

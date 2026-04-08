import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless Ubuntu environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Task 1: Dataset Preparation ---
print("Step 1: Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# --- Task 2: Data Preprocessing ---
print("Step 2: Standardizing features...")
# PCA is distance-based; we must scale data so one feature doesn't dominate others
X_std = StandardScaler().fit_transform(X)

# --- Task 3 & 4: Applying PCA and Variance Analysis ---
print("Step 3: Performing PCA analysis...")
pca = PCA()
X_pca = pca.fit_transform(X_std)

# Calculate explained variance
exp_var = pca.explained_variance_ratio_
print(f"Explained Variance by component: {exp_var}")

# Plotting the Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(exp_var) + 1), exp_var, marker='o', linestyle='--', color='b')
plt.title('Scree Plot: Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Variance Ratio')
plt.grid(True)
plt.savefig('pca_variance_plot.png')
print("✓ Scree plot saved as 'pca_variance_plot.png'")

# --- Task 5: Classification Using Reduced Dimensions ---
print("\nStep 4: Training Classifier on top 2 components...")
# We choose the first 2 components because they usually contain ~95% of the info
X_pca_2 = X_pca[:, :2]

# Split the reduced dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca_2, y, test_size=0.3, random_state=42)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f"Original Number of Features: {X.shape[1]}")
print(f"Reduced Number of Features:  {X_pca_2.shape[1]}")
print(f"Classification Accuracy:     {accuracy:.4f}")
print("-" * 40)

# Final Visual: Scatter Plot of the 2 Principal Components
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca_2[y == i, 0], X_pca_2[y == i, 1], label=target_name)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Iris Dataset Projected onto First Two PCs')
plt.legend()
plt.savefig('pca_clusters.png')
print("✓ Cluster visualization saved as 'pca_clusters.png'")

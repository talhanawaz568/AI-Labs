import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

# --- Task 1: Load and Split the Dataset ---
print("Task 1: Loading Iris Dataset...")
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Define features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# 1.2: Perform a standard Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✓ Training data shape: {X_train.shape}")
print(f"✓ Testing data shape: {X_test.shape}")

# --- Task 2: Implement k-Fold Cross-Validation ---
print("\nTask 2: Initializing Logistic Regression and performing 5-Fold CV...")
# We increase max_iter to 200 to ensure the solver converges
model = LogisticRegression(max_iter=200)

# 2.2: Perform 5-fold cross-validation
# This splits the data into 5 groups and rotates the test set 5 times
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"✓ Individual CV scores: {cv_scores}")
print(f"✓ Mean CV Score: {cv_scores.mean():.4f}")

# --- Task 3: Compare Performance ---
print("\nTask 3: Comparing with simple Train-Test Split...")

# Train on the 80% split
model.fit(X_train, y_train)

# Evaluate on the 20% test set
test_score = model.score(X_test, y_test)

print("-" * 40)
print(f"Simple Train-Test Score: {test_score:.4f}")
print(f"5-Fold Cross-Val Mean:  {cv_scores.mean():.4f}")
print("-" * 40)

# --- Conclusion Analysis ---
print("\n[Analysis for Lab Report]")
if abs(test_score - cv_scores.mean()) < 0.05:
    print("Conclusion: The model is stable. Both evaluation methods yield similar results.")
else:
    print("Conclusion: High variance detected. Cross-validation provides a more reliable estimate.")

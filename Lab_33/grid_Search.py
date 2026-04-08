import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# --- Task 2: Load and Prepare Dataset ---
print("Step 1: Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Task 1: Define the Parameter Grid ---
# We are testing 4 values of C, 4 values of gamma, and 2 kernels.
# Total combinations = 4 * 4 * 2 = 32 different models to train!
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# --- Task 2: Setup and Run GridSearchCV ---
print("Step 2: Starting GridSearchCV (Exhaustive Search)...")
# refit=True ensures the best model is saved and ready for predictions
# verbose=1 shows a summary of the search progress
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)

# This will run the training 32 times (times the number of Cross-Validation folds)
grid.fit(X_train, y_train)

# --- Task 3: Display Results and Evaluation ---
print("\n" + "="*40)
print("RESULTS")
print("="*40)
print(f"Best Parameters found: {grid.best_params_}")
print(f"Best Score during training: {grid.best_score_:.4f}")

print("\nStep 3: Evaluating on unseen test data...")
grid_predictions = grid.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, grid_predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, grid_predictions))

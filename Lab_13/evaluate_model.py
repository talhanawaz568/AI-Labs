import matplotlib
# Force Matplotlib to use 'Agg' backend for headless Ubuntu
matplotlib.use('Agg') 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# --- Task 1: Generate a Confusion Matrix ---
print("Task 1: Loading Dataset and Training Random Forest...")
data = load_iris()
X, y = data.data, data.target

# Split the dataset (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict results
y_pred = clf.predict(X_test)

# Generate Confusion Matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

# Visualize using ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap='viridis', ax=ax)
plt.title('Confusion Matrix: Iris Classification')

# Save the plot
output_image = 'confusion_matrix.png'
plt.savefig(output_image)
print(f"✓ Confusion Matrix saved as '{output_image}'")

# --- Task 2: Calculate Precision, Recall, and F1-Score ---
print("\nTask 2: Generating Classification Report...")
report = classification_report(y_test, y_pred, target_names=data.target_names)
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(report)
print("="*60)

# --- Task 3: Interpretation ---
print("\nTask 3: Brief Analysis")
print("-" * 30)
accuracy = (np.diag(cm).sum() / cm.sum()) * 100
print(f"Overall Model Accuracy: {accuracy:.2f}%")
print("Analysis: The diagonal elements in the saved image show the correct predictions.")
print("If off-diagonal elements are 0, the model achieved perfect classification for that test set.")

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
import graphviz
import os

# --- Task 1: Train a Decision Tree ---
print("Task 1: Loading Iris dataset and training model...")
iris = load_iris()
X, y = iris.data, iris.target

# Initialize and Train (Using a max_depth to keep the visualization clean)
clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X, y)

accuracy = clf.score(X, y)
print(f"✓ Model Training Complete. Accuracy: {accuracy:.2f}")

# --- Task 2: Visualize the Tree Structure ---
print("\nTask 2.1: Text-Based Representation:")
text_representation = export_text(clf, feature_names=list(iris.feature_names))
print(text_representation)

print("\nTask 2.2: Generating Graphical Visualization...")
# Create DOT data
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=iris.feature_names,  
                           class_names=list(iris.target_names),
                           filled=True, rounded=True,  
                           special_characters=True)  

# Render the graph to a file
graph = graphviz.Source(dot_data)  
output_path = graph.render("iris_decision_tree", format="png") 
print(f"✓ Visual Tree saved as: {output_path}")

# --- Task 3: Analyze Feature Importances ---
print("\nTask 3: Feature Importance Analysis:")
feature_importances = clf.feature_importances_
for feature, importance in zip(iris.feature_names, feature_importances):
    print(f" - {feature:20}: {importance:.4f}")

print("\n--- Lab 12 Complete ---")

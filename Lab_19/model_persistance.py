import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

# --- Part 1: Training and Saving (The "Developer" Phase) ---
print("Part 1: Training the model...")
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to disk
model_filename = 'iris_rf_model.joblib'
joblib.dump(model, model_filename)
print(f"✓ Model successfully saved to: {model_filename}")

# Clean up the original model variable to prove we are using the saved file
del model

# --- Part 2: Loading and Predicting (The "Production" Phase) ---
print("\nPart 2: Loading the model for production use...")

if os.path.exists(model_filename):
    # Load the model back into memory
    loaded_model = joblib.load(model_filename)
    print("✓ Model loaded back into memory.")

    # Select a sample from the test set for prediction
    sample_index = 0
    sample_data = X_test[sample_index].reshape(1, -1)
    actual_label = y_test[sample_index]

    # Make a prediction
    predicted_class = loaded_model.predict(sample_data)
    
    # Map the numeric prediction back to flower names
    predicted_name = iris.target_names[predicted_class][0]
    actual_name = iris.target_names[actual_label]

    print("\n" + "="*40)
    print(f"SAMPLE DATA: {X_test[sample_index]}")
    print(f"PREDICTION:  {predicted_name} (Class {predicted_class[0]})")
    print(f"ACTUAL:      {actual_name} (Class {actual_label})")
    print("="*40)
else:
    print("Error: Model file not found!")

print("\n--- Lab 19 Complete ---")

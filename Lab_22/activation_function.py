import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Suppress TensorFlow warnings for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Task 1: Load and Prepare the Dataset ---
print("Preparing Iris Dataset...")
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the target (e.g., 2 becomes [0, 0, 1])
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- Define the Model Building Function ---
def build_model(activation_type):
    model = Sequential()
    
    # Handle Leaky ReLU differently as it is a Layer, not a string
    if activation_type == 'leaky_relu':
        model.add(Dense(64, input_shape=(4,)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.01))
    else:
        model.add(Dense(64, input_shape=(4,), activation=activation_type))
        model.add(Dense(64, activation=activation_type))
    
    # Final layer is always Softmax for 3-class classification
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# --- Task 2: Compare Training Results ---
results = []
activation_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu']

print("\nStarting comparative training...")

for func in activation_functions:
    print(f" > Training with: {func.upper()}")
    model = build_model(func)
    
    # verbose=0 keeps the console clean during the 50 epochs
    model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=0)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    results.append({'Activation': func, 'Loss': loss, 'Accuracy': accuracy})

# --- Task 3: Summary Report ---
df_results = pd.DataFrame(results)

print("\n" + "="*45)
print("FINAL COMPARISON TABLE")
print("="*45)
print(df_results.to_string(index=False))
print("="*45)

print("\nObservation Notes:")
print("- ReLU and Leaky ReLU usually reach high accuracy faster.")
print("- Sigmoid can be 'slower' because its gradients are very small at the edges.")
print("- Tanh is zero-centered, often making it perform better than Sigmoid.")

import numpy as np
import matplotlib
# Force Agg backend for headless Ubuntu
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- Task 1.1: Simulating a Single Neuron ---
def simulate_neuron(inputs, weights, bias):
    # output = sum(weights * inputs) + bias
    weighted_sum = np.dot(inputs, weights) + bias
    return weighted_sum

# Define sample inputs (e.g., features of a flower) and weights (importance)
inputs = np.array([1.5, 2.0, 0.5])
weights = np.array([0.8, -0.5, 1.2])
bias = 0.1

raw_output = simulate_neuron(inputs, weights, bias)
print(f"1. Raw Neuron Output (Weighted Sum): {raw_output:.4f}")

# --- Task 1.2: Activation Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Apply functions to our raw output
print(f"2. Output after Sigmoid (Probability): {sigmoid(raw_output):.4f}")
print(f"3. Output after ReLU: {relu(raw_output):.4f}")

# --- Task 2: Visualizing Activation Functions ---
print("\nGenerating visualization for Lab Report...")
x_vals = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 5))

# Plot Sigmoid
plt.subplot(1, 2, 1)
plt.plot(x_vals, sigmoid(x_vals), color='blue', linewidth=2)
plt.title("Sigmoid Function (Binary Logic)")
plt.grid(True, alpha=0.3)

# Plot ReLU
plt.subplot(1, 2, 2)
plt.plot(x_vals, relu(x_vals), color='red', linewidth=2)
plt.title("ReLU Function (Hidden Layer Logic)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions.png')
print("✓ Visualization saved as 'activation_functions.png'")

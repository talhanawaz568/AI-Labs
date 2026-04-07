import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os

# Set logging level to suppress unnecessary TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Task 2: Load and Preprocess the Dataset ---
print("Task 2.1: Loading MNIST dataset...")
mnist = tf.keras.datasets.mnist

# Load 60,000 training images and 10,000 test images
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize: Convert pixel values from 0-255 to 0-1 for faster convergence
x_train, x_test = x_train / 255.0, x_test / 255.0

# --- Task 1: Build the Sequential Model ---
print("\nTask 1.3: Building the MLP Model...")
model = Sequential([
    # Flattens the 28x28 image into a 1D vector of 784 pixels
    Flatten(input_shape=(28, 28)), 
    
    # Hidden layer with 128 neurons using ReLU activation
    Dense(128, activation='relu'), 
    
    # Output layer with 10 neurons (one for each digit 0-9) 
    # Softmax turns raw scores into probabilities
    Dense(10, activation='softmax') 
])

# --- Task 2.2: Compile the Model ---
# Adam is a popular optimizer that automatically adjusts the learning rate
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Task 2.3: Train the Model ---
print("\nTask 2.3: Starting training for 5 Epochs...")
model.fit(x_train, y_train, epochs=5, batch_size=32)

# --- Task 3: Evaluate Model Accuracy ---
print("\nTask 3.1: Evaluating on unseen test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print("\n" + "="*40)
print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
print("="*40)

# Optional: Save the model for future use (Lab 19 concept)
model.save('mnist_mlp_model.h5')
print("\n✓ Model saved as 'mnist_mlp_model.h5'")

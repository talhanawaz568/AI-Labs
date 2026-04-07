import os
import numpy as np
import matplotlib
# Force Agg backend for headless environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Task 2: Explore the MNIST Dataset ---
print("Task 2: Loading and Inspecting MNIST Dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

print(f'Training data shape: {train_images.shape}')
print(f'Test data shape: {test_images.shape}')

# Save an example image for inspection
plt.figure(figsize=(5, 5))
plt.imshow(train_images[0], cmap='gray')
plt.title(f'Sample Label: {train_labels[0]}')
plt.savefig('mnist_sample_image.png')
print("✓ Sample image saved as 'mnist_sample_image.png'")

# --- Task 3: Data Preprocessing ---
print("\nTask 3: Preprocessing data...")
# Normalize pixel values to [0, 1]
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Reshape to (60000, 28, 28, 1) - The '1' represents the grayscale channel
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# --- Model Building ---
print("Building CNN Model...")
model = models.Sequential([
    # Convolutional Layer: Learns 32 different spatial filters/patterns
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # Pooling Layer: Downsamples the image to reduce computation and noise
    layers.MaxPooling2D((2, 2)),
    
    # Flattening: Convert 2D feature maps into a 1D vector
    layers.Flatten(),
    
    # Fully Connected Layers: Use the patterns to perform the classification
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# --- Compile and Train ---
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("\nStarting Training (5 Epochs)...")
model.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=1)

# --- Evaluate ---
print("\nEvaluating Model on Test Data...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("\n" + "="*40)
print(f"CNN TEST ACCURACY: {test_acc:.4f}")
print("="*40)

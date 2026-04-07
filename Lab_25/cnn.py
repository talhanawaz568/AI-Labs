import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Suppress TensorFlow logging for clarity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Task 1.2: Load and Pre-process the Dataset ---
print("Task 1.2: Loading and Reshaping MNIST Data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing pixel values to [0, 1] - Crucial for CNN convergence
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape data to (Samples, Width, Height, Channels)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# One-hot encode target labels (e.g., 3 becomes [0,0,0,1,0,0,0,0,0,0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# --- Task 1.3: Build the CNN Model ---
print("\nTask 1.3: Building the CNN Model...")
model = Sequential()

# Feature Extraction: Conv2D learns 32 different filters
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Down-sampling: Reduces spatial dimensions by 50% to prevent overfitting
model.add(MaxPooling2D(pool_size=(2, 2)))

# Transition: Flattens the 2D maps into a 1D vector for the Dense layer
model.add(Flatten())

# Classification: Mapping extracted features to the 10 digit classes
model.add(Dense(10, activation='softmax'))

# --- Task 2: Train the CNN ---
print("\nTask 2: Compiling and Training...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training for 3 epochs as specified
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=128)

# --- Task 3: Evaluate and Analyze ---
print("\nTask 3: Evaluating Performance...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print("-" * 45)
print(f'FINAL TEST LOSS:     {loss:.4f}')
print(f'FINAL TEST ACCURACY: {accuracy:.4f}')
print("-" * 45)

# --- Task 3.2: Analysis of Potential Improvements ---
print("\n[Discussion & Improvements]")
print("1. Depth: Adding another Conv2D/MaxPooling layer could capture more complex features.")
print("2. Dropout: Adding a Dropout layer would help if the model shows signs of overfitting.")
print("3. Epochs: Increasing epochs might reach higher accuracy, but risks overfitting.")
print("4. Augmentation: Rotating or shifting images would make the model more robust.")

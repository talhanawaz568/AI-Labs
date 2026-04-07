import numpy as np
import os
import matplotlib
# Force Agg backend for headless Ubuntu
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Task 1.2: Load and Preprocess MNIST Data ---
print("Loading and preprocessing MNIST data...")
(X_train, y_train), (X_val, y_val) = tf.keras.datasets.mnist.load_data()

# Flatten 28x28 images to 784 vectors and normalize to [0, 1]
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
X_val = X_val.reshape(X_val.shape[0], -1).astype('float32') / 255

# One-hot encode labels (e.g., 5 becomes [0,0,0,0,0,1,0,0,0,0])
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)

# Define Early Stopping Callback
# patience=3 means stop if validation loss doesn't improve for 3 straight epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# --- Task 1.3: Model WITHOUT Regularization ---
print("\nTraining Model 1: Baseline (No L2)...")
model = Sequential([
    Dense(512, input_dim=784, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=50, batch_size=200, 
                    callbacks=[early_stopping], verbose=1)

# --- Task 2: Model WITH L2 Regularization ---
print("\nTraining Model 2: With L2 Regularization (0.01)...")
model_reg = Sequential([
    # kernel_regularizer adds a penalty for large weights
    Dense(512, input_dim=784, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(10, activation='softmax')
])

model_reg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_reg = model_reg.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=50, batch_size=200, 
                        callbacks=[early_stopping], verbose=1)

# --- Task 3: Visualize Results ---
print("\nGenerating performance comparison plot...")
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_loss'], label='Baseline (No L2)')
plt.plot(history_reg.history['val_loss'], label='With L2 Regularization')
plt.title('Validation Loss: Effect of L2 Regularization')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

output_plot = 'regularization_comparison.png'
plt.savefig(output_plot)
print(f"✓ Analysis plot saved as '{output_plot}'")

# Final Comparison
print("\n" + "="*40)
print(f"Baseline Epochs: {len(history.history['loss'])}")
print(f"L2 Reg.  Epochs: {len(history_reg.history['loss'])}")
print("="*40)

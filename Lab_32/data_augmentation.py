import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

img_path = 'sample_image.jpg'

# Pre-check: Is the file actually an image?
try:
    with Image.open(img_path) as test_img:
        test_img.verify() 
    print("✓ Image file verified successfully.")
except Exception as e:
    print(f"X Error: The file {img_path} is invalid or corrupted.")
    print("Try running: wget -O sample_image.jpg https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg")
    exit()

# If verified, proceed with loading
print("Loading and processing image...")
img = load_img(img_path)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

# ... (rest of your augmentation code)
datagen = ImageDataGenerator(rotation_range=40, horizontal_flip=True, vertical_flip=True)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(2, 2, i + 1)
    plt.imshow(array_to_img(batch[0]))
    i += 1
    if i == 4: break

plt.savefig('augmented_results.png')
print("✓ Results saved to augmented_results.png")



## wget -O sample_image.jpg https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg
# pip install scipy tensorflow matplotlib

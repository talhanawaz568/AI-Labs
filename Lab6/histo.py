import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Step 3.1: Prepare Histogram Data
data = np.random.randn(1000)

# Step 3.2: Plot the Histogram
plt.hist(data, bins=30, color='green', alpha=0.7, edgecolor='white')

# Step 3.3: Customize
plt.xlabel('Data Value')
plt.ylabel('Frequency')
plt.title('Task 3: Histogram (Normal Distribution)')

# Save result
plt.savefig('histogram.png')
print("Task 3 Complete: saved as 'histogram.png'")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Step 2.1: Prepare Data
categories = ['A', 'B', 'C', 'D']
values = [4, 7, 1, 8]

# Step 2.2: Plot the Bar Chart
plt.bar(categories, values, color='lightblue', edgecolor='black')

# Step 2.3: Customize
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Task 2: Bar Chart Example')

# Save result
plt.savefig('bar_chart.png')
print("Task 2 Complete: saved as 'bar_chart.png'")

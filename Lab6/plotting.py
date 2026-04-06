import matplotlib.pyplot as plt

# 1. Force Matplotlib to use the 'Agg' backend 
# (This tells it to generate files instead of opening a window)
import matplotlib
matplotlib.use('Agg') 

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y, label='Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.legend()

# 2. CHANGE THIS: Instead of plt.show(), use plt.savefig()
plt.savefig('my_plot.png')
print("Plot saved successfully as 'my_plot.png'")

import numpy as np
arr1 = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr1)


arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:")
print(arr2)

element = arr1[2]
print("Accessed Element:", element)

subarray = arr2[0:2, 1:3]
print("Sliced Array:")
print(subarray)

reshaped_array = arr1.reshape(5, 1)
print("Reshaped Array:")
print(reshaped_array)

mean_value = np.mean(arr1)
print("Mean:", mean_value)

total_sum = np.sum(arr2)
print("Total Sum:", total_sum)

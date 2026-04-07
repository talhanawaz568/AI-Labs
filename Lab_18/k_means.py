import matplotlib
# Force Agg backend for headless Ubuntu
matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import os

# --- Task 1: Generate and Visualize Sample Dataset ---
print("Task 1: Generating synthetic blobs...")
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.7)
plt.title("Task 1: Raw Sample Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig('cluster_task1_raw.png')
print("✓ Raw dataset visualization saved.")

# --- Task 2: Apply Elbow Method ---
print("\nTask 2.1: Running Elbow Method...")
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0, n_init='auto')
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', color='purple')
plt.title("Task 2.1: The Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.savefig('cluster_task2_elbow.png')
print("✓ Elbow plot saved. Look for the 'bend' in the line.")

# --- Task 3: Apply K-Means and Visualize Clusters ---
print("\nTask 3: Final Clustering with k=4...")
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0, n_init='auto')
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'cyan']
for i in range(4):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')

# Plotting Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=250, c='yellow', marker='X', edgecolors='black', label='Centroids')

plt.title("Task 3: K-Means Final Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig('cluster_task3_final.png')

print(f"\n--- Lab 18 Complete ---")
print(f"Results saved in: {os.getcwd()}")

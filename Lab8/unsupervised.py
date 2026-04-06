import matplotlib
# Force headless mode for Ubuntu CLI
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Generate sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. Initialize k-Means with 4 clusters
print("Running Unsupervised Clustering (k-Means)...")
kmeans = KMeans(n_clusters=4, n_init='auto')

# 3. Fit and predict the cluster labels
y_kmeans = kmeans.fit_predict(X)

# 4. Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Data Points')

# Mark the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')

plt.title('k-Means Clustering Results')
plt.legend()

# 5. Save the plot as an image
output_file = 'clustering_output.png'
plt.savefig(output_file)
print(f"✓ Task Complete. Plot saved as '{output_file}'")

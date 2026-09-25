import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
X = np.array([
    [1, 2],
    [1, 3],
    [2, 2],
    [8, 8],
    [9, 8],
    [8, 9],
    [4, 5],
    [5, 5],
    [5, 6]
])

# K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train 
kmeans.fit(X)

labels = kmeans.labels_

centroids = kmeans.cluster_centers_

print("Cluster Labels:")
print(labels)

print("\nCentroids:")
print(centroids)

# # Visualize 
plt.scatter(X[:, 0], X[:, 1], c=labels)

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show()
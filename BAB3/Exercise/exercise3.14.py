# Exercise 3.14: Add two points to each K-Means group
import numpy as np
from sklearn.cluster import KMeans

X = np.array([
    [1, 2, 3], [1, 4, 2], [1, 0, 3],
    [2, 3, 2], [2, 1, 3],
    [10, 2, 4], [9, 4, 3], [11, 0, 2],
    [10, 3, 3], [12, 1, 2],
])

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
print('Cluster labels:', kmeans.labels_)
print('Cluster centers:', kmeans.cluster_centers_)
print('Predicted cluster:', kmeans.predict([[12, 3, 1]]))

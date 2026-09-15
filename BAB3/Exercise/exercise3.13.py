# Exercise 3.13 K-means Clustering with generated data
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.0,
    random_state=0,
)

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
print('Cluster labels:', kmeans.labels_)
print('Cluster centers:', kmeans.cluster_centers_)
print('Predicted cluster:', kmeans.predict([[0, 0]]))

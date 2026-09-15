# Exercise 3.7: Principal Component Analysis Breast Cancer
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load Breast Cancer data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Standardize features before applying PCA.
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the first two original features.
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Mean radius')
plt.ylabel('Mean texture')
plt.title('Original Breast Cancer Data')

# Plot the data in the first two PCA components.
plt.figure(2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Breast Cancer PCA Data')
plt.show()

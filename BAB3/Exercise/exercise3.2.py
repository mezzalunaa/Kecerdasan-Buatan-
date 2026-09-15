"""Exercise 3.2: SVM Iris classification using petal features."""
from sklearn import datasets, svm

iris = datasets.load_iris()

# Use the third and fourth features: petal length and petal width.
X = iris.data[:, 2:4]
y = iris.target  # 0: Setosa, 1: Versicolour, 2: Virginica
print(y)

clf = svm.SVC()
clf.fit(X, y)

# Predict a flower from its petal length and width.
p = clf.predict([[1.5, 0.4]])
print(p)

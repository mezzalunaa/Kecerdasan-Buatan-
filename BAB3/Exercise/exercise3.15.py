# Exercise 3.15: Semi-supervised learning with three groups
import numpy as np
from sklearn.semi_supervised import LabelSpreading

X = np.array([
    [0, 1], [1, 1], [2, 0], [3, 1],
    [10, 5], [11, 6], [12, 4], [13, 5],
    [20, 10], [21, 11], [22, 9], [23, 10],
])

# Only one point in each group is labeled initially.
labels = np.full(len(X), -1.0)
labels[0] = 0
labels[4] = 1
labels[8] = 2

label_spread = LabelSpreading(kernel='knn', n_neighbors=3, alpha=0.8)
label_spread.fit(X, labels)
print('Initial labels:', labels)
print('Output labels:', label_spread.transduction_)

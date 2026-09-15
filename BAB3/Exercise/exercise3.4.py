"""Exercise 3.4 exercise: histograms of selected breast cancer features."""
from matplotlib import pyplot
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# The first four features are radius, texture, perimeter (size), and smoothness.
features = cancer.data[:, :4]
feature_names = [
    'Mean radius',
    'Mean texture',
    'Mean perimeter (size)',
    'Mean smoothness',
]

figure, axes = pyplot.subplots(2, 2, figsize=(10, 7))
for axis, values, name in zip(axes.ravel(), features.T, feature_names):
    axis.hist(values, bins=20, edgecolor='black')
    axis.set_title(name)
    axis.set_xlabel('Value')
    axis.set_ylabel('Number of data points')

figure.suptitle('Breast Cancer Feature Histograms')
figure.tight_layout()
pyplot.show()

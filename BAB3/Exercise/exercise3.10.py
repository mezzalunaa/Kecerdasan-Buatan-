# Exercise 3.10 Comparison of Classifiers on Diabetes data
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    'SVM',
    'Naive Bayes',
    'LDA',
    'QDA',
    'Decision Tree',
    'Random Forest',
    'Nearest Neighbors',
    'Neural Networks',
]
classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=1000, random_state=0),
]

X, target = load_diabetes(return_X_y=True)
# Convert the continuous target into low and high disease-progression classes.
y = (target > np.median(target)).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name + ': ' + str(score))

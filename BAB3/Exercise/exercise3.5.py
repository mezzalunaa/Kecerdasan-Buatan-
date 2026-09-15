# Exercise 3.5 exercise: save and load a Naive Bayes Iris model
import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)

# Train the model.
clf = GaussianNB()
clf.fit(X, y)

# Save the trained model to a file.
model_file = Path(__file__).with_name('naive_bayes_iris.pkl')
with model_file.open('wb') as file:
    pickle.dump(clf, file)

# Load the model from the file.
with model_file.open('rb') as file:
    loaded_clf = pickle.load(file)

# Predict with the loaded model.
p = loaded_clf.predict([[5.0, 3.4, 1.5, 0.4]])
print(p)

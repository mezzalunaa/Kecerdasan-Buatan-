# Example 3.28C Plot LazyPredict classification accuracy
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)
classifier = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = classifier.fit(X_train, X_test, y_train, y_test)

plt.figure(figsize=(10, 5))
plt.plot(models.index, models['Accuracy'])
plt.xticks(rotation=90)
plt.ylabel('Accuracy')
plt.title('LazyPredict Classification Accuracy')
plt.tight_layout()
plt.show()

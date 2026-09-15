# Exercise 3.20 LazyPredict classification on Wine data
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

try:
    from lazypredict.Supervised import LazyClassifier
except ImportError:
    print('Install LazyPredict first with: pip install lazypredict')
else:
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    classifier = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = classifier.fit(X_train, X_test, y_train, y_test)
    print(models)

    plt.figure(figsize=(10, 5))
    plt.plot(models.index, models['Accuracy'], '-s')
    plt.xticks(rotation=90)
    plt.ylabel('Accuracy')
    plt.title('Wine Classification Accuracy')
    plt.tight_layout()
    plt.show()

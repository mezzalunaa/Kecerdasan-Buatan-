# Examples 3.28B-E LazyPredict classification and regression
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split

try:
    from lazypredict.Supervised import LazyClassifier, LazyRegressor
except ImportError:
    print('Install LazyPredict first with: pip install lazypredict')
else:
    # Classification on Iris data.
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )
    classifier = LazyClassifier(verbose=0, ignore_warnings=True)
    classification_models, classification_predictions = classifier.fit(
        X_train, X_test, y_train, y_test
    )
    print('Classification models:')
    print(classification_models)

    plt.figure(figsize=(10, 5))
    plt.plot(classification_models.index, classification_models['Accuracy'], '-s')
    plt.xticks(rotation=90)
    plt.ylabel('Accuracy')
    plt.title('LazyPredict Classification Accuracy')
    plt.tight_layout()
    plt.show()

    # Regression on California Housing data.
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )
    regressor = LazyRegressor(verbose=0, ignore_warnings=True)
    regression_models, regression_predictions = regressor.fit(
        X_train, X_test, y_train, y_test
    )
    print('Regression models:')
    print(regression_models)

    plt.figure(figsize=(10, 5))
    plt.plot(regression_models.index, regression_models['R-Squared'], '-s')
    plt.xticks(rotation=90)
    plt.ylabel('R-Squared')
    plt.title('LazyPredict Regression R-Squared')
    plt.tight_layout()
    plt.show()

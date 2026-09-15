# Exercise 3.21 LazyPredict regression on Diabetes data
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

try:
    from lazypredict.Supervised import LazyRegressor
except ImportError:
    print('Install LazyPredict first with: pip install lazypredict')
else:
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )
    regressor = LazyRegressor(verbose=0, ignore_warnings=True)
    models, predictions = regressor.fit(X_train, X_test, y_train, y_test)
    print(models)

    plt.figure(figsize=(10, 5))
    plt.plot(models.index, models['R-Squared'], '-s')
    plt.xticks(rotation=90)
    plt.ylabel('R-Squared')
    plt.title('Diabetes Regression R-Squared')
    plt.tight_layout()
    plt.show()

# Exercise 3.18 Regression on California Housing data
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=1
)
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1, n_estimators=100)
reg3 = LinearRegression()
reg4 = MLPRegressor(random_state=1, max_iter=1000)
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3), ('NN', reg4)])
ereg.fit(X_train, y_train)

predictions = ereg.predict(X_test[:20])
print('VotingRegressor predictions:', predictions)
print('Test R-squared:', ereg.score(X_test, y_test))
plt.plot(predictions, 'r*', label='VotingRegressor')
plt.xlabel('Test samples')
plt.ylabel('Predicted house value')
plt.title('California Housing Regression')
plt.legend()
plt.show()

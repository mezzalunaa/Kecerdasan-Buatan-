# Exercise 3.12 Multiple Linear Regression on Linnerud data
from sklearn import linear_model
from sklearn.datasets import load_linnerud

x, y = load_linnerud(return_X_y=True)

reg = linear_model.LinearRegression()
reg.fit(x, y)
print('Coefficients:\n', reg.coef_)
print('Intercept:\n', reg.intercept_)
pred = reg.predict([x[0]])
print('Prediction for the first sample:\n', pred)

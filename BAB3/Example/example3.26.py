# Example 3.26 AutoML / Automatic Model Selection (Modern Alternative)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("[INFO] Menjalankan alternatif AutoML untuk California Housing Dataset...")

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Mean Squared Error (MSE) : {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2 Score                 : {r2_score(y_test, y_pred):.4f}")
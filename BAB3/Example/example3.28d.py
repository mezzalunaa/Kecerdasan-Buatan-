# Example 3.28D LazyPredict regression (Optimized)
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

print("[INFO] Mengunduh data dan menyiapkan LazyRegressor...")

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X_sample = X.sample(n=1000, random_state=1)
y_sample = y.loc[X_sample.index]

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.1, random_state=1
)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print("\n--- HASIL EVALUASI MODEL LAZYPREDICT ---")
print(models)
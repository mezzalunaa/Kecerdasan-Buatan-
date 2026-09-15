# Example 3.28E Plot LazyPredict regression R-squared (Fixed)
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def main():
    print("[INFO] Mengunduh dataset dan menyiapkan sampel data...")
    
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X_sample = X.sample(n=1000, random_state=1)
    y_sample = y.loc[X_sample.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.1, random_state=1
    )

    print("[INFO] Melatih puluhan model regresi...")
    regressor = LazyRegressor(verbose=0, ignore_warnings=True)
    models, predictions = regressor.fit(X_train, X_test, y_train, y_test)

    print("[INFO] Menampilkan grafik perbandingan R-Squared...")
    plt.figure(figsize=(10, 5))
    plt.plot(models.index, models['R-Squared'], '-s', color='steelblue', linewidth=1.5)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel('R-Squared')
    plt.title('LazyPredict Regression R-Squared Comparison')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
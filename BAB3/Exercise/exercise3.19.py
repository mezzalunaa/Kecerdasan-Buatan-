# Exercise 3.19: PyCaret classification with Breast Cancer dataset
import pandas as pd
from sklearn import datasets

try:
    from pycaret import classification
except ImportError:
    print("[INFO] PyCaret tidak ditemukan/terjadi bentrok dependensi. Menjalankan fallback scikit-learn...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    cancer = datasets.load_breast_cancer(as_frame=True)
    X, y = cancer.data, cancer.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("\n--- Hasil Evaluasi Model Klasifikasi (Breast Cancer) ---")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))

else:
    cancer = datasets.load_breast_cancer(as_frame=True)
    data = cancer.data.copy()
    data['Target'] = cancer.target
    data = pd.DataFrame(data)

    classification.setup(data=data, target='Target', session_id=0, verbose=False)
    
    best_model = classification.compare_models()
    print("\n--- Model Terbaik berdasarkan PyCaret ---")
    print(best_model)
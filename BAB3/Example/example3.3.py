# Example 3.3 Python SVM Iris CSV Classifications
import os
import pandas as pd
from sklearn import svm, datasets

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'iris.csv')

if not os.path.exists(csv_path):
    print("[INFO] File 'iris.csv' tidak ditemukan. Membuat file otomatis...")
    iris = datasets.load_iris()
    df_temp = pd.DataFrame(iris.data[:, :2], columns=['sepal_length', 'sepal_width'])
    df_temp['species'] = [iris.target_names[i] for i in iris.target]
    df_temp.to_csv(csv_path, index=False)
    print(f"[INFO] File 'iris.csv' berhasil dibuat di: {csv_path}")

df = pd.read_csv(csv_path)
X = df.values[:, :2]
s = df['species']

d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

clf = svm.SVC()
clf.fit(X, y)

p = clf.predict([[5.4, 3.2]])
print("Hasil Prediksi (p):", p)
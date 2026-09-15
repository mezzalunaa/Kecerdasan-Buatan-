# Example 3.27 PyCaret classification
import pandas as pd
from sklearn import datasets

try:
    from pycaret import classification
except ImportError:
    print('Example 3.27 requires the optional pycaret package.')
else:
    iris = datasets.load_iris(as_frame=True)
    data = iris.data.copy()
    data['Target'] = iris.target
    data = pd.DataFrame(data)
    classification.setup(data=data, target='Target', session_id=0, verbose=False)
    print(classification.compare_models())

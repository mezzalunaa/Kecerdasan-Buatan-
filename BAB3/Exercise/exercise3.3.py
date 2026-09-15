"""Exercise 3.3 exercise: scatter plot of Iris sepal features."""
import pandas as pd
from matplotlib import pyplot

url = (
	'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/'
	'0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
)
df = pd.read_csv(url).dropna()

colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species, group in df.groupby('species'):
	pyplot.scatter(
		group['sepal_length'],
		group['sepal_width'],
		color=colors[species],
		label=species,
	)

pyplot.xlabel('Sepal length (cm)')
pyplot.ylabel('Sepal width (cm)')
pyplot.title('Iris Sepal Length and Width')
pyplot.legend()
pyplot.show()

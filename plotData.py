from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#Import the iris dataset
dataFile = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print dataFile.tail()

#assign the y and X matrixes
labels = dataFile.iloc[0:100,4].values
weights = dataFile.iloc[0:100, [0, 2]].values

#Change labeling from text to the integer
labels = np.where(labels == 'Iris-setosa', -1, 1)


#plot it baby!
plt.scatter(weights[:50, 0], weights[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(weights[50:100, 0], weights[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()
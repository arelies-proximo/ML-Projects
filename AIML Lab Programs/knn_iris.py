import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset =  pd.read_csv('knn_iris.csv')
x = dataset.iloc[:,:-1].values
    #all rows, all columns except last
y = dataset.iloc[:,4].values
    #all rows, last column containing target

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.20)

plt.plot(xtrain, ytrain,'b.', xtest, ytest, 'r.')
    #b. blue and r. red


classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(xtrain, ytrain)

accuracy0 = classifier.score(xtest, ytest)
accuracy1 = classifier.score(xtrain, ytrain)

print("Accuracy0: ",accuracy0)
print("Accuracy1: ",accuracy1)

example = np.array([5.7, 3, 4.2, 1.2])
print(example)
example = example.reshape(1,-1)
print(example)

pred = classifier.predict(example)

print(pred)

plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()

x = pd.DataFrame(iris.data)
x.columns = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width']
    #assign column names

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

model = KMeans(n_clusters=3)
model.fit(x)

colormap = np.array(['red', 'blue', 'lime'])

plt.subplot(2,2,1)
plt.scatter(x.Petal_length, x.Petal_width, color=colormap[y.Targets], s=40)
plt.title('Real Clusters')
plt.xlabel('Petal length')
plt.ylabel('Petal width')


plt.subplot(2,2,2)
plt.scatter(x.Petal_length, x.Petal_width, color=colormap[model.labels_], s=40)
plt.title('KMean Clusterings')
plt.xlabel('Petal length')
plt.ylabel('Petal width')


from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

scaler = preprocessing.StandardScaler()
scaler.fit(x)

xsa = scaler.transform(x)
xs = pd.DataFrame(xsa, columns=x.columns)

gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y = gmm.predict(xs)

plt.subplot(2,2,3)
plt.scatter(x.Petal_length, x.Petal_width, color=colormap[gmm_y], s=40)
plt.title('GMM Clusterings')
plt.xlabel('Petal length')
plt.ylabel('Petal width')


plt.show()
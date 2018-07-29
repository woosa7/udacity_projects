import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np

X,y = datasets.make_moons(n_samples=1500, noise=.05)

x1 = X[:,0]
x2 = X[:,1]

plt.scatter(x1,x2,s=5)
plt.show()

#results with DBSCAN algorithm
dbscan = DBSCAN(eps=0.1)
dbscan.fit(X)
y_pred = dbscan.labels_.astype(np.int)

colors = np.array(['#ff0000', '#00ff00'])

plt.title('clusters with DBSCAN')
plt.scatter(x1,x2,s=5,color=colors[y_pred])
plt.show()

#results with K-Means Clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.labels_.astype(np.int)

colors = np.array(['#ff0000', '#00ff00'])

plt.title('clusters with K-Means')
plt.scatter(x1,x2,s=5,color=colors[y_pred])
plt.show()

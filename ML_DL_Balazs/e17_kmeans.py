import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


x, y = make_blobs(n_samples=100, centers=5, random_state=0, cluster_std=0.6)
plt.scatter(x[:,0], x[:,1], s=50)
plt.show()

est = KMeans(5)
est.fit(x)
y_cluster = est.predict(x)
print(y_cluster)

plt.scatter(x[:,0], x[:,1], c=y_cluster, s=50, cmap='rainbow')
plt.show()


#----------------------------------------------------------------
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x, y)

array = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
kmeans = KMeans(n_clusters=2)
kmeans.fit(array)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print('-----------')
print(centroids)
print(labels)

colors = ["g.", "r."]

for i in range(len(x)):
    print("Coordinate: ", array[i], " label: ", labels[i])
    plt.plot(array[i][0], array[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidth=5, zorder=10)
plt.show()


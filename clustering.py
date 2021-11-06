from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, ward
import mglearn



# X, y = make_blobs(random_state=1)
X, y = make_moons(n_samples=20, noise=0.1)
s = MinMaxScaler()
X = s.fit_transform(X)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()
a = AgglomerativeClustering(n_clusters=2, linkage="complete")
a.fit(X)

linkage_a = ward(X)

dendrogram(linkage_a)
plt.title("Dendrogram")
plt.show()





# print(y)
# print(km.labels_)
# print(km.cluster_centers_)
# db = DBSCAN(min_samples=5, eps=0.3)
# clusters = db.fit_predict(X)

algorithms = [KMeans(n_clusters=2),
              AgglomerativeClustering(n_clusters=2),
              DBSCAN(min_samples=5, eps=0.3),
              DBSCAN(min_samples=3, eps=0.2),
              ]

fig, axes = plt.subplots(1, 4, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(15, 3))

for axis, algor in zip(axes, algorithms):
    clusters = algor.fit_predict(X)
    axis.scatter(X[:, 0], X[:, 1], clusters)
    axis.set_title("{}, ARI: {:.2f}".format(algor.__class__.__name__, adjusted_rand_score(y, clusters)))

plt.show()




mglearn.discrete_scatter(X[:, 0], X[:, 1], clusters, markers="o")
# mglearn.discrete_scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], [0, 1], markers="^", markeredgewidth=2)
#
plt.show()


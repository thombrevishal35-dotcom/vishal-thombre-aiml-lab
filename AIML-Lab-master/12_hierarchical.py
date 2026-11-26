from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

data = load_iris()
X = StandardScaler().fit_transform(data.data)

# Dendrogram
linked = linkage(X, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode="lastp", p=30)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample / Cluster"); plt.ylabel("Distance")
plt.show()

hier = AgglomerativeClustering(n_clusters=3).fit_predict(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
print("Hierarchical silhouette:", silhouette_score(X, hier))
print("K-Means silhouette:", silhouette_score(X, kmeans))

plt.scatter(X[:, 0], X[:, 1], c=hier)
plt.title("Hierarchical Clusters (Iris)")
plt.xlabel(data.feature_names[0]); plt.ylabel(data.feature_names[1])
plt.show()

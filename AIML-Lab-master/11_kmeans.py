from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data = load_iris()
X = StandardScaler().fit_transform(data.data)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)
print("Silhouette score:", silhouette_score(X, labels))

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel(data.feature_names[0]); plt.ylabel(data.feature_names[1])
plt.title("K-Means Clustering (Iris)")
plt.show()

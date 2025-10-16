import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
iris = load_iris()
x = iris.data[:, :2]
y = iris.target
sil_scores = []
for k in range(2, 11):
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(x)
sil_scores.append(silhouette_score(x, labels))
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

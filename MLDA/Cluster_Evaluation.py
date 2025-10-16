from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,
davies_bouldin_score, calinski_harabasz_score
X, y_true = make_blobs(n_samples=500, n_features=2, centers=4,
cluster_std=1.0, random_state=42)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)
silhouette = silhouette_score(X, y_pred)
davies_bouldin = davies_bouldin_score(X, y_pred)
calinski_harabasz = calinski_harabasz_score(X, y_pred)
print('Cluster Evaluation Results:')
print(f'→ Silhouette Score: {silhouette:.3f} of k:{k} (Higher is better,
range: -1 to 1)')
print(f'→ Davies-Bouldin Index: {davies_bouldin:.3f} (Lower is
better)')
print(f'→ Calinski-Harabasz Index: {calinski_harabasz:.3f} (Higher
is better)')

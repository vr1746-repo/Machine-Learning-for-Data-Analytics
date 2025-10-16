import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
iris = load_iris()
y = iris.target
scaler = StandardScaler()
x_scaled = scaler.fit_transform(iris.data)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
plt.figure()
plt.title(&#39;PCA of Iris Dataset&#39;)
plt.xlabel(&#39;Principal Component 1&#39;)
plt.ylabel(&#39;Principal Component 2&#39;)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.legend()
plt.show()

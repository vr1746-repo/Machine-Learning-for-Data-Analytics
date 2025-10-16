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
plt.title("PCA of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.legend()
plt.show()

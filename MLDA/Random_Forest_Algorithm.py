import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
x, y = iris.data, iris.target
model = RandomForestClassifier(n_estimators=100,
random_state=42)
model.fit(x, y)
feature_importance = model.feature_importances_
feature_names = iris.feature_names
plt.figure(figsize=(8, 5))
plt.barh(feature_names, feature_importance)
plt.show()

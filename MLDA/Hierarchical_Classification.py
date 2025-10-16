import numpy s np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
iris = load_iris()
x = iris.data[:, :2]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=42)
hc_model = AgglomerativeClustering(n_clusters=3, linkage="ward")
y_hc = hc_model.fit_predict(x_test)
print(&quot;Classification Report:\n&quot;, classification_report(y_test, y_hc))
cm = confusion_matrix(y_test, y_hc)
print(&quot;Confusion Matrix:\n&quot;, cm)

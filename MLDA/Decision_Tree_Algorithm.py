import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
iris = load_iris()
x, y = iris.data, iris.target
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x, y)
plt.figure(figsize=(10, 5))
plot_tree(model, feature_names=iris.feature_names,
class_names=iris.target_names)
plt.title('Decision Tree')
plt.show()

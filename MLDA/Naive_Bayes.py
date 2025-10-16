import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
iris = load_iris()
y = iris.target
x = iris.data[:, :2]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=21)
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(&quot;Classification Report:\n&quot;, classification_report(y_test,
y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n";, cm)

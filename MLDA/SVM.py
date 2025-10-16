import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=21)
model = SVC(kernel=&#39;linear&#39;)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Classification Report:\n";, classification_report(y_test,
y_pred))
cm = confusion_matrix(y_test, y_pred)
print(&quot;Confusion Matrix:\n&quot;, cm)

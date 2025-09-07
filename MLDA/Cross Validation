from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

# Load dataset
X, y = datasets.load_iris(return_X_y=True)

# Models
clf1 = DecisionTreeClassifier(random_state=42)
clf2 = LogisticRegression(max_iter=1000)

# 10-fold cross validation
k_folds = KFold(n_splits=10)

# Cross-validation scores
scores1 = cross_val_score(clf1, X, y, cv=k_folds)
scores2 = cross_val_score(clf2, X, y, cv=k_folds)

# Results
print("Decision Tree CV Scores:", scores1)
print("Logistic Regression CV Scores:", scores2)

print("Decision Tree Average CV Score:", scores1.mean())
print("Logistic Regression Average CV Score:", scores2.mean())

print("Number of CV Scores (Decision Tree):", len(scores1))
print("Number of CV Scores (Logistic Regression):", len(scores2))

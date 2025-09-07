import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

data = pd.read_csv("Employee_Details.csv")

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

X = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]
y = data["left"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)
y_dt_pred = dt.predict(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_lr_pred = lr.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_dt_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_lr_pred))


table = [[0, 0], [0, 0]]

for i in range(len(y_test)):
    if y_test.iloc[i] == y_dt_pred[i] and y_test.iloc[i] != y_lr_pred[i]:
        table[0][1] += 1   # DT correct, LR wrong
    elif y_test.iloc[i] != y_dt_pred[i] and y_test.iloc[i] == y_lr_pred[i]:
        table[1][0] += 1   # LR correct, DT wrong

    elif y_test.iloc[i] == y_dt_pred[i] and y_test.iloc[i] == y_lr_pred[i]:
        table[0][0] += 1   # Both correct
    else:
        table[1][1] += 1   # Both wrong

print("\nContingency Table (McNemar):")
print(table)

result = mcnemar(table, exact=True)  # exact=True recommended for small samples
print("\nMcNemar’s Test Statistic:", result.statistic)
print("McNemar’s Test p-value:", result.pvalue)

if result.pvalue < 0.05:
    print("➡ Significant difference between models (Reject H0)")
else:
    print("➡ No significant difference between models (Fail to reject H0)")

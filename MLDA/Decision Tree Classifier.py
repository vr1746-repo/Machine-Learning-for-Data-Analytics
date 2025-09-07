import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Employee_Details.csv")
print("First few rows of dataset:")
print(data.head())

le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
print("\nAfter Label Encoding 'salary':")
print(data.head())

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

print("\nFeatures (X) sample:")
print(X.head())
print("\nTarget (y) sample:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, y_train)

y_dt_pred = dt.predict(X_test)

print("\ny_test shape:", y_test.shape)
print("y_dt_pred shape:", y_dt_pred.shape)

accuracy_dt = accuracy_score(y_test, y_dt_pred)
print("\nDecision Tree Accuracy:", accuracy_dt)

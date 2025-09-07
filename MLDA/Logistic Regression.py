import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_lr_pred = lr.predict(X_test)

print("\ny_test shape:", y_test.shape)
print("y_lr_pred shape:", y_lr_pred.shape)

accuracy_lr = accuracy_score(y_test, y_lr_pred)
print("\nLogistic Regression Accuracy:", accuracy_lr)

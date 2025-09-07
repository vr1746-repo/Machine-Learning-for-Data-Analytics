import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

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

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

dt = DecisionTreeClassifier(criterion="entropy", random_state=2)
dt.fit(x_train, y_train)
y_dt_pred = dt.predict(x_test)

accuracy_dt = accuracy_score(y_test, y_dt_pred)
print("\nDecision Tree Classifier Accuracy:", accuracy_dt)

lrr = LinearRegression()
lrr.fit(x_train, y_train)
y_lrr_pred = lrr.predict(x_test)

mse = mean_squared_error(y_test, y_lrr_pred)
r2 = r2_score(y_test, y_lrr_pred)

print("Linear Regression MSE:", mse)
print("Linear Regression R²:", r2)

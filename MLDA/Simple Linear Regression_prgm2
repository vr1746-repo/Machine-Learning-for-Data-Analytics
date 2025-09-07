import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("HousingData.csv", encoding="latin1")
print("Original Data:")
print(data.head())

# Data Cleaning: remove commas and convert to float
for col in ["Size (sqft)", "Age (years)", "Price(INR)"]:
    data[col] = data[col].replace(",", "", regex=True).astype(float)

# Define features and target
X = data[["Size (sqft)", "Rooms", "Age (years)", "Location"]]
y = data["Price(INR)"]

# Numerical and categorical features
num_features = ["Size (sqft)", "Rooms", "Age (years)"]
cat_features = ["Location"]

# Pipelines for preprocessing
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

# Final pipeline with Linear Regression
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Print regression coefficients
reg = model.named_steps["regressor"]
print("\nModel Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Evaluation
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Compare actual vs predicted
results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print("\nActual vs Predicted:")
print(results.head())

# Plot actual vs predicted
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")
plt.title("Actual vs Predicted House Prices")
plt.show()

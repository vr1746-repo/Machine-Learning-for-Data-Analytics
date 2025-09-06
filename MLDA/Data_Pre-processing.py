import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("HousingData.csv", encoding="latin1")

# Display the first few rows of the dataset
print("Original Data:")
print(data.head())

# 1. Data Cleaning
for col in ["Size (sqft)", "Age (years)", "Price(INR)"]:
    data[col] = data[col].replace(",", "", regex=True).astype(float)

# Numerical features
num_features = ["Size (sqft)", "Rooms", "Age (years)", "Price(INR)"]

num_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]
)

# Categorical features
cat_features = ["Location"]

cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)

# Combine preprocessing steps of Input Data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, ["Size (sqft)", "Rooms", "Age (years)"]),
        ("cat", cat_transformer, cat_features),
    ]
)

preprocessor.set_output(transform="pandas")

# Apply the transformations to the Input data
data_preprocessed = preprocessor.fit_transform(data)
print("\nPreprocessed Input Data:")
print(data_preprocessed.head())

# Combine preprocessing steps of Output Data
preprocessor_Out = ColumnTransformer(
    transformers=[
        ("num", num_transformer, ["Price(INR)"])
    ]
)

preprocessor_Out.set_output(transform="pandas")

# Apply the transformations to the Output data
data_preprocessed_Out = preprocessor_Out.fit_transform(data)
print("\nPreprocessed Output Data:")
print(data_preprocessed_Out.head())

# 2. Feature Engineering
data_preprocessed["Price_per_sqft"] = data["Price(INR)"] / data["Size (sqft)"]

# 3. Data Splitting
X = data_preprocessed
y = data_preprocessed_Out

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display the first few rows of the processed training data
print("\nX_train:")
print(X_train.head())
print("\ny_train:")
print(y_train.head())

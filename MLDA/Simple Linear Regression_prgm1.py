import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
data = {'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 5, 4, 5]}

df = pd.DataFrame(data)

# Independent (X) and dependent (Y) variables
X = df[['x']]
Y = df[['y']]

# Create and fit the model
model = LinearRegression()
model.fit(X, Y)

# Predictions
y_pred = model.predict(X)

# Print slope and intercept
print("Slope (m):", model.coef_[0][0])
print("Intercept (c):", model.intercept_[0])

# Plot data and regression line
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # Features
y = iris.target # Labels

scaler = StandardScaler()
X = scaler.fit_transform(X)
# Convert target to categorical (for multi-class classification)
y = keras.utils.to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
model = keras.Sequential([
layers.Dense(8, activation='relu', input_shape=(X.shape[1],)), #
Input + 1st hidden layer
layers.Dense(8, activation='relu'), # 2nd
hidden layer
layers.Dense(3, activation='softmax') # Output
layer (3 classes)
])

model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=8,
verbose=1)
# Step 6: Make predictions
predictions = model.predict(X_test)
print('\nSample Predictions (first 5 samples):')
print(np.round(predictions[:5],3))

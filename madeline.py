import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Encode target labels (0, 1, 2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize random weights for a hidden layer (64 neurons)
hidden_layer_weights = np.random.randn(X_train.shape[1], 64)  # Random weights (4 features x 64 neurons)
hidden_layer_bias = np.random.randn(64)  # Random bias for each neuron

# Apply ReLU activation to the hidden layer outputs
X_train_hidden = relu(np.dot(X_train, hidden_layer_weights) + hidden_layer_bias)
X_test_hidden = relu(np.dot(X_test, hidden_layer_weights) + hidden_layer_bias)

# Perform Linear Regression on the transformed data
regressor = LinearRegression()
regressor.fit(X_train_hidden, y_train)

# Predict using the trained model
y_pred = regressor.predict(X_test_hidden)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"MADALINE Regression MSE (Alternative Approach): {mse:.4f}")

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.datasets import load_iris

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the ADALINE model using SGDRegressor
adaline = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
adaline.fit(X_train, y_train)

# Make predictions
y_pred_linear = adaline.predict(X_test)
y_pred_sigmoid = sigmoid(y_pred_linear)  # Apply sigmoid activation to predictions

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred_sigmoid)

print(f"ADALINE Regression MSE (with Sigmoid Activation): {mse:.4f}")

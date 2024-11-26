import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load Iris dataset
data = load_iris()
X = data.data[:, 0].reshape(-1, 1)  # Use the first feature for simplicity
y = data.target.reshape(-1, 1)  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Explore polynomial degrees from 1 to 9
degrees = range(1, 10)
train_errors = []
test_errors = []

for degree in degrees:
    # Polynomial feature transformation
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # SGD Regressor
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    sgd_reg.fit(X_train_poly, y_train.ravel())

    # Predictions and errors
    y_train_pred = sgd_reg.predict(X_train_poly)
    y_test_pred = sgd_reg.predict(X_test_poly)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# Plot the errors for different polynomial degrees
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label="Train Error", marker='o')
plt.plot(degrees, test_errors, label="Test Error", marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Polynomial Degree vs. Error")
plt.legend()
plt.grid(True)
plt.show()

# Best degree conclusion
best_degree = degrees[np.argmin(test_errors)]
print(f"The best polynomial degree is {best_degree} with test error {min(test_errors):.2f}.")

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Adaline:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])  # Initialize weights
        self.bias = 0  # Initialize bias
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                y_pred = np.dot(xi, self.weights) + self.bias
                error = target - y_pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

    def predict(self, X):
        return (np.dot(X, self.weights) + self.bias >= 0).astype(int)

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate the Adaline model
model = Adaline(lr=0.01, epochs=1000)
model.fit(X_train, y_train)
accuracy = np.mean(model.predict(X_test) == y_test)

print("Accuracy:", accuracy)

import numpy as np

# Define the Boolean expression ((p ∧ q) ∨ r) → (p ∧ ∼r)
def boolean_expression(p, q, r):
    lhs = (p and q) or r
    rhs = p and not r
    return not lhs or rhs

# Generate data for the Boolean expression
data = []
for p in [0, 1]:
    for q in [0, 1]:
        for r in [0, 1]:
            output = boolean_expression(p, q, r)
            data.append([p, q, r, output])

data = np.array(data)
X = data[:, :-1]  # Features
Y = data[:, -1]   # Labels

# Perceptron implementation
class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=1000):
        self.weights = np.zeros(input_size)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero
        self.lr = lr  # Learning rate
        self.epochs = epochs  # Number of training epochs

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = self.activation_function(linear_output)
                error = target - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation_function(output) for output in linear_output])

# Train and test the perceptron
perceptron = Perceptron(input_size=3)
perceptron.fit(X, Y)
predictions = perceptron.predict(X)
accuracy = np.mean(predictions == Y)

print(f"Perceptron Accuracy: {accuracy * 100:.2f}%")

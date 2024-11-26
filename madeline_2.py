from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.datasets import load_iris

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights and biases for the hidden layer
hidden_layer_weights = np.random.randn(X_train.shape[1], 64)  # Random weights
hidden_layer_bias = np.random.randn(64)  # Random biases

# Compute the hidden layer outputs using ReLU activation
X_train_hidden = relu(np.dot(X_train, hidden_layer_weights) + hidden_layer_bias)
X_test_hidden = relu(np.dot(X_test, hidden_layer_weights) + hidden_layer_bias)

# Scale the hidden layer outputs for better training of the logistic regression model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_hidden)
X_test_scaled = scaler.transform(X_test_hidden)

# Train a logistic regression classifier on the transformed data
classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)  # Increased max_iter for convergence
classifier.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy and classification report
print(f"\nMADALINE Classification Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

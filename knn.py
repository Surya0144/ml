import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

def minkowski_distance(x1, x2, p=3):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

def k_nearest_neighbors(X_train, y_train, X_test, k, p=3):
    y_pred = []
    for x_test in X_test:
        distances = [minkowski_distance(x_test, x_train, p) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 5
y_pred = k_nearest_neighbors(X_train, y_train, X_test, k, p=3)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of k-NN classifier (k={k}, p=3): {accuracy * 100:.2f}%")

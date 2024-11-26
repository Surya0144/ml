import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load dataset (Iris dataset in this case)
data = load_iris()
X = data.data  # Features

# Standardize the data (important for K-means performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# List to store inertia values for different K
inertia_values = []

# Try different values of K (number of clusters)
K_range = range(1, 11)  # Try K from 1 to 10 clusters
for k in K_range:
    # Initialize KMeans model with k clusters
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X_scaled)
    # Append the inertia (sum of squared distances) to the list
    inertia_values.append(kmeans.inertia_)

# Plot Inertia vs. Number of Clusters (K)
plt.figure(figsize=(8, 6))
plt.plot(K_range, inertia_values, marker='o', linestyle='-', color='b')
plt.title('Inertia vs. Number of Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

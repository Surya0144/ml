import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # True labels (not used in clustering)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Agglomerative Clustering
clustering = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
y_pred = clustering.fit_predict(X_scaled)

# Plot the dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Calculate the silhouette score
sil_score = silhouette_score(X_scaled, y_pred)
print(f"Silhouette Score: {sil_score:.2f}")
 
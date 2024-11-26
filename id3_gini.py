from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import numpy as np 
data=load_iris() 
X=data.data 
y=data.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Train Decision Tree using Gini criterion 
dt = DecisionTreeClassifier(criterion='gini', random_state=42) 
dt.fit(X_train, y_train) 
# Predict on test data 
y_pred = dt.predict(X_test) 
# Evaluate accuracy 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Decision Tree (Gini) Accuracy: {accuracy*100:.2f}%")
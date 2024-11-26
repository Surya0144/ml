from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_iris 
data = load_iris() 
X=data.data 
y=data.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
dt = DecisionTreeClassifier(criterion='entropy') 
dt.fit(X_train, y_train) 
y_pred = dt.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Decision Tree (Entropy) Accuracy: {accuracy*100:.2f}%") 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_iris 
data = load_iris() 
X = data.data   
y = data.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
nb = GaussianNB() 
nb.fit(X_train, y_train) 
y_pred = nb.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Naïve Bayes Classifier Accuracy: {accuracy * 100:.2f}%") 
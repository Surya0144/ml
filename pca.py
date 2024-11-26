from sklearn.decomposition import PCA 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
data = load_iris() 
X = data.data  
y = data.target  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
log_reg = LogisticRegression(max_iter=200) 
log_reg.fit(X_train, y_train) 
y_pred_before = log_reg.predict(X_test) 
accuracy_before = accuracy_score(y_test, y_pred_before) 
pca = PCA(n_components=3)  
X_train_pca = pca.fit_transform(X_train) 
X_test_pca = pca.transform(X_test) 
log_reg.fit(X_train_pca, y_train) 
y_pred_after = log_reg.predict(X_test_pca) 
accuracy_after = accuracy_score(y_test, y_pred_after) 
print(f"Logistic Regression Accuracy (before PCA): {accuracy_before * 100:.2f}%") 
print(f"Logistic Regression Accuracy (after PCA): {accuracy_after * 100:.2f}%")
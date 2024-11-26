import numpy as np 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
data = load_iris() 
X = data.data 
y = data.target  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
log_reg = LogisticRegression(max_iter=1000, random_state=42) 
log_reg.fit(X_train, y_train) 
y_pred_no_lda = log_reg.predict(X_test) 
accuracy_no_lda = accuracy_score(y_test, y_pred_no_lda) 
print("Logistic Regression Accuracy without LDA:", accuracy_no_lda) 
lda = LinearDiscriminantAnalysis(n_components=2)   
X_train_lda = lda.fit_transform(X_train, y_train) 
X_test_lda = lda.transform(X_test) 
log_reg_lda = LogisticRegression(max_iter=1000, random_state=42) 
log_reg_lda.fit(X_train_lda, y_train) 
y_pred_lda = log_reg_lda.predict(X_test_lda) 
accuracy_with_lda = accuracy_score(y_test, y_pred_lda) 
print("Logistic Regression Accuracy with LDA:", accuracy_with_lda) 
print("\nClassification Report without LDA:") 
print(classification_report(y_test, y_pred_no_lda)) 
print("\nClassification Report with LDA:") 
print(classification_report(y_test, y_pred_lda))
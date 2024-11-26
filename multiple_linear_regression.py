from sklearn.linear_model import SGDRegressor 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.datasets import load_iris 
data=load_iris() 
X=data.data 
y=data.target 
label_encoder = LabelEncoder() 
y = label_encoder.fit_transform(y) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42) 
sgd.fit(X_train, y_train) 
y_pred = sgd.predict(X_test) 
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
accuracy_score = accuracy_score(y_test,y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}") 
print(f"R-squared (R2) Score: {r2:.4f}") 
print(f"accuracy score : {accuracy_score:.2f}")
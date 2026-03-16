                   #Breast Cancer Classification using Machine Learning

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load dataset
data=load_breast_cancer()
X, y = data.data, data.target

#Split dataset
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Feature Scaling
scaler=StandardScaler()
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)

#Train model
model=LogisticRegression(max_iter=5000)
model.fit(X_train_s, y_train)

#Predictions
y_pred=model.predict(X_test_s)

#Evaluation
print("Accuracy:",accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
                   #Breast Cancer Classification using Machine Learning
#1.Import libraries
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

#2.Load dataset
data=load_breast_cancer()
X = data.data
y = data.target


print("Dataset loaded")
print("Shape:", X.shape)

#3.Train-TestSplit dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)

#4.Feature Scaling
scaler=StandardScaler()
X_train_s=scaler.fit_transform(X_train) #FIT + TRANSFORM
X_test_s=scaler.transform(X_test) #ONLY TRANSFORM

print("Scaling completed")

#5.Model Training
base_model=LogisticRegression(max_iter=1000)
model=CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
model.fit(X_train_s, y_train)

print("Model trained")

#6.Predictions
y_pred=model.predict(X_test_s)

#7.Evaluation
print("\n Model Performance:")

print("Accuracy:",accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#8.Save model
joblib.dump(model,"models/model.pkl")
print("\n Model saved in models/model.pkl")

#9.Save scaler
joblib.dump(scaler,"models/scaler.pkl")
print("Scaler saved in models/scaler.pkl")
import joblib
import numpy as np

#1.Load model and scaler
model=joblib.load("models/model.pkl")
scaler=joblib.load("models/scaler.pkl")

print("Model and scaler loaded")

#2.Example input(30 features)
sample=np.random.rand(1, 30)

#Apply scaling (IMPORTANT)
sample_scaled=scaler.transform(sample)

#Predict  
prediction=model.predict(sample_scaled)

print("Prediction (0=malignant, 1=benign):", prediction[0])
import joblib
import numpy as np

#1.Load model
model=joblib.load("models/model.pkl")

print("Model loaded")

#2.Example input(30 features)
sample=np.random.rand(1, 30)

prediction=model.predict(sample)

print("Prediction (0=malignant, 1=benign):", prediction[0])
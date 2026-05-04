from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

#Create a FastAPI app
app = FastAPI()

#Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Load model
model=joblib.load("models/model.pkl")
scaler=joblib.load("models/scaler.pkl")

#Define input format
class InputData(BaseModel):
    features: list[float]

#Home route
@app.get("/")
def home():
    return {"message": "Breast Cancer FASTAPI is running"}

#Prediction route
@app.post("/predict")
def predict(data: InputData):
    #step1: Validate input length
    if len(data.features) != 30:
        return {"error": "Exactly 30 features required"}
    #convert to numpy array
    features = np.array(data.features).reshape(1, -1)
    #scale features
    features_scaled = scaler.transform(features)
    #Get probabilities
    proba = model.predict_proba(features_scaled)[0]
    #Determine prediction
    prediction = int(proba[1] > 0.5)
    #Get confidence
    confidence = float(proba[prediction])
    #Step2: Clamp confidence (avoid 100%)
    confidence = min(confidence, 0.999)
    return {"prediction": prediction, "result": "Benign" if prediction == 1 else "Malignant", "confidence": (round(confidence * 100, 2)}

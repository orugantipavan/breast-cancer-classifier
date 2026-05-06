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
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float

    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

#Home route
@app.get("/")
def home():
    return {"message": "Breast Cancer FASTAPI is running"}

#Prediction route
@app.post("/predict")
def predict(data: InputData):
    #Step 1: Convert input ordered list(VERY IMPORTANT ORDER)
    features_list = [data.mean_radius, data.mean_texture, data.mean_perimeter, data.mean_area, data.mean_smoothness, data.mean_compactness, data.mean_concavity, data.mean_concave_points, data.mean_symmetry, data.mean_fractal_dimension, data.radius_error, data.texture_error, data.perimeter_error, data.area_error, data.smoothness_error, data.compactness_error, data.concavity_error, data.concave_points_error, data.symmetry_error, data.fractal_dimension_error, data.worst_radius, data.worst_texture, data.worst_perimeter, data.worst_area, data.worst_smoothness, data.worst_compactness, data.worst_concavity, data.worst_concave_points, data.worst_symmetry, data.worst_fractal_dimension]
    
    print("RAW INPUT:", features_list)
    #Step 2: Convert to numpy array
    features = np.array(features_list).reshape(1, -1)
    #step 3: Scale features
    features_scaled = scaler.transform(features)
    print("SCALED INPUT:", features_scaled)
    #Step 4: Predict probability
    proba = model.predict_proba(features_scaled)[0]
    print("PROBABILIETIES:", proba)
    #Step 5: Decision
    prediction = int(proba[1] > 0.5)
    #Get confidence
    confidence = float(proba[prediction])
    #Step 6: Clamp confidence (avoid 100%)
    confidence = min(confidence, 0.999)
    return {"prediction": prediction, "result": "Benign" if prediction == 1 else "Malignant", "confidence": round(confidence * 100, 2)}

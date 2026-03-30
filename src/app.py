from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

#Initailize app
app = Flask(__name__)
CORS(app)

#Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("Model and scaler loaded")

#Home route
@app.route("/")
def home():
    return "Breast Cancer Prediction API is running"

#Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        #Expecting input as list of 30 features
        features = np.array(data["features"]).reshape(1, -1)

        #Scale input
        features_scaled = scaler.transform(features)
        
        #Predict
        prediction = model.predict(features_scaled)[0]

        result = "Benign" if prediction == 1 else "Malignant"

        return jsonify({"prediction": int(prediction), "result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
#Run server
if __name__ == "__main__":
    app.run(debug=True)
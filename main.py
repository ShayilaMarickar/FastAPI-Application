from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="House Price Prediction API")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class PredictionInput(BaseModel):
    bedrooms: float
    bathrooms: float
    stories: float

class PredictionOutput(BaseModel):
    prediction: float
    confidence_interval: List[float]
@app.get("/")
def health_check():
    return{"status": "Healthy", "message": "House Price Prediction API is Running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features = np.predict([[input_data.bedrooms, input_data.bathrooms, input_data.stories]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        confidence_std = 0.1 * prediction
        confidence_interval = [float(prediction - confidence_std), float(prediction + confidence_std)]

        return PredictionOutput(
            prediction = float(prediction),
            confidence_interval=confidence_interval
        )    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")
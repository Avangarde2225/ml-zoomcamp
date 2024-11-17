# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging


class PredictionRequest(BaseModel):
    # Replace these with your actual feature names and types
    DISTANCE: float
    TOTAL_CARGO: float
    CARRIER_GROUP_NEW: int
    UNIQUE_CARRIER_encoded: int
    Org_Dest_Country_FE: float
    MONTH_SIN: float
    MONTH_COS: float
    CLASS_F: int
    CLASS_G: int
    CLASS_L: int
    CLASS_P: int
    REGION_A: int
    REGION_D: int
    REGION_I: int
    REGION_L: int
    REGION_P: int
    REGION_S: int
    IS_PEAK_SEASON: int

app = FastAPI()

# Load the model
model = joblib.load('best_model.pkl')



@app.get("/")
async def read_root():
    return {"message": "Passenger Prediction App is running!"}

logging.basicConfig(level=logging.INFO)
@app.post('/predict')
async def predict(request: PredictionRequest):
    logging.info(f"Received request: {request.dict()}")
    try:
        input_data = pd.DataFrame([request.dict()])
        feature_order = [
            'DISTANCE', 'TOTAL_CARGO', 'CARRIER_GROUP_NEW', 'UNIQUE_CARRIER_encoded',
            'Org_Dest_Country_FE', 'MONTH_SIN', 'MONTH_COS', 'CLASS_F', 'CLASS_G',
            'CLASS_L', 'CLASS_P', 'REGION_A', 'REGION_D', 'REGION_I', 'REGION_L',
            'REGION_P', 'REGION_S', 'IS_PEAK_SEASON'
        ]
        input_data = input_data[feature_order]
        prediction = model.predict(input_data)
        return {'prediction': prediction.tolist()}
    except Exception as e:
        return {'error': str(e)}
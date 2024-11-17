import joblib
import pandas as pd

# Load the model
model = joblib.load('best_model.pkl')

# Define test input
input_data = pd.DataFrame([{
    "DISTANCE": 500000.0,
    "TOTAL_CARGO": 200000.0,
    "CARRIER_GROUP_NEW": 1,
    "UNIQUE_CARRIER_encoded": 15,
    "Org_Dest_Country_FE": 0.75,
    "MONTH_SIN": 0.5,
    "MONTH_COS": 0.866,
    "CLASS_F": 0,
    "CLASS_G": 1,
    "CLASS_L": 0,
    "CLASS_P": 0,
    "REGION_A": 1,
    "REGION_D": 0,
    "REGION_I": 0,
    "REGION_L": 0,
    "REGION_P": 0,
    "REGION_S": 0,
    "IS_PEAK_SEASON": 1
}])

# Make prediction
prediction = model.predict(input_data)

print("Prediction:", prediction)
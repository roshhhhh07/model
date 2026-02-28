from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# ✅ DEFINE SCHEMA FIRST
class TransportInput(BaseModel):
    Route_ID: str
    Stop_ID: str
    Urban_Zone: str
    Route_Type: str
    Boarding_Count: int
    Alighting_Count: int
    Total_Pax: int
    Avg_Speed: float
    Year: int
    Month: int
    DayOfWeek: str
    Is_Weekend: int
    Season: str

# ✅ LOAD MODEL + ENCODERS
model = joblib.load("congestion_model.pkl")
encoders = joblib.load("encoders.pkl")

# ✅ ENDPOINT AFTER SCHEMA
@app.post("/predict_congestion")
def predict(data: TransportInput):

    df = pd.DataFrame([data.dict()])

    categorical_cols = [
        "Route_ID",
        "Stop_ID",
        "Route_Type",
        "Urban_Zone",
        "Season",
        "DayOfWeek"
    ]

    for col in categorical_cols:
        df[col] = encoders[col].transform(df[col])

    pred = model.predict(df)[0]

    return {"Congestion_Index": float(pred)}

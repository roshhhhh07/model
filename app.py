from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Transport ML API")

model = joblib.load("congestion_model.pkl")
encoders = joblib.load("encoders.pkl")

@app.get("/")
def home():
    return {"status": "Transport ML API Live"}

@app.post("/predict_congestion")
def predict(data: dict):

    df = pd.DataFrame([data])

    # encode categorical
    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    X = df[[
        "Boarding_Count","Alighting_Count","Total_Pax",
        "Avg_Speed","Route_Type","Urban_Zone",
        "DayOfWeek","Is_Weekend","Season"
    ]]

    pred = model.predict(X)[0]

    return {"Congestion_Index": float(pred)}
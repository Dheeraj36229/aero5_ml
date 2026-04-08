from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("filtration_model.pkl")

class PredictRequest(BaseModel):
    aqi: float
    breathing: float
    activity: int

@app.post("/predict")
def predict(data: PredictRequest):
    
    aqi = data.aqi
    breathing = data.breathing
    activity = data.activity

    input_data = np.array([[aqi, breathing, activity]])
    protection = float(model.predict(input_data)[0])

    penalty = max(0, (breathing - 15) * 1.2 + activity * 5)
    penalty = min(penalty, 30)

    effective_filtration = protection - penalty
    final_filtration = max(20, min(95, effective_filtration))

    if penalty > 20:
        warning = "High breathing stress"
        advice = "Reduce activity or rest"
    else:
        warning = "Normal"
        advice = "Safe"

    return {
        "protection_raw": f"{round(protection, 2)}%",
        "breathing_penalty": f"{round(penalty, 2)}%",
        "effective_filtration": f"{round(final_filtration, 2)}%",
        "warning": warning,
        "advice": advice
    }
@app.get("/predict")
def predict_get(aqi: float, breathing: float, activity: int):

    input_data = np.array([[aqi, breathing, activity]])
    protection = float(model.predict(input_data)[0])

    penalty = max(0, (breathing - 15) * 1.2 + activity * 5)
    penalty = min(penalty, 30)

    effective_filtration = protection - penalty
    final_filtration = max(20, min(95, effective_filtration))

    if penalty > 20:
        warning = "High breathing stress"
        advice = "Reduce activity or rest"
    else:
        warning = "Normal"
        advice = "Safe"

    return {
        "protection_raw": f"{round(protection, 2)}%",
        "breathing_penalty": f"{round(penalty, 2)}%",
        "effective_filtration": f"{round(final_filtration, 2)}%",
        "warning": warning,
        "advice": advice
    }
    

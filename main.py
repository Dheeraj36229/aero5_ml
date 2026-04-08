from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# ✅ CORS (important for app/webview)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Lazy load model (VERY IMPORTANT for Render)
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = joblib.load("filtration_model.pkl")
        print("Model loaded")

# ✅ Request model (for POST)
class PredictRequest(BaseModel):
    aqi: float
    breathing: float
    activity: int

# ✅ Root route (to test server)
@app.get("/")
def home():
    return {"status": "AERO5 API running"}

# ✅ POST endpoint (for app)
@app.post("/predict")
def predict(data: PredictRequest):
    
    load_model()

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

# ✅ GET endpoint (for browser / fallback)
@app.get("/predict")
def predict_get(aqi: float, breathing: float, activity: int):

    load_model()

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

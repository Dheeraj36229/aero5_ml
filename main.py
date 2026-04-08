from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# ✅ CORS (important for app/webview)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Lazy model loading (safe for deployment)
model = None

def load_model():
    global model
    if model is None:
        try:
            print("Loading model...")
            model = joblib.load("filtration_model.pkl")
            print("Model loaded successfully")
        except Exception as e:
            print("MODEL ERROR:", e)
            raise e

# ✅ Request schema
class PredictRequest(BaseModel):
    aqi: float
    breathing: float
    activity: int

# ✅ Root route (health check)
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "AERO5 API running"}

# ✅ Extra ping route (use for uptime monitoring)
@app.get("/ping")
def ping():
    return {"status": "alive"}

# ✅ POST endpoint (main usage)
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

# ✅ GET endpoint (for browser testing)
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

# ✅ For Railway / Render dynamic port handling
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("filtration_model.pkl")

@app.post("/predict")
def predict(aqi: float, breathing: float, activity: int):
    
    # Step 1: ML prediction
    input_data = np.array([[aqi, breathing, activity]])
    protection = float(model.predict(input_data)[0])

    # Step 2: Calculate breathing penalty
    penalty = max(0, (breathing - 15) * 1.2 + activity * 5)
    penalty = min(penalty, 30)

    # Step 3: Effective filtration
    effective_filtration = protection - penalty

    # Step 4: Clamp output
    final_filtration = max(20, min(95, effective_filtration))

    # Step 5: Message
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
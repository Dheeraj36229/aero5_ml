import joblib

# Load trained model
model = joblib.load("filtration_model.pkl")

# Example test input
# Format: [AQI, BreathingRate, Activity]
test_data = [[20, 22, 2]]

# Predict
prediction = model.predict(test_data)

print("Predicted Filtration:", round(prediction[0], 2), "%")
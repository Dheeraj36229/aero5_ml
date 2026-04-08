import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# STEP 1: Load dataset
data = pd.read_csv("data.csv")
print(data.head())  # Display first few rows to verify data loading

# STEP 2: Separate features and target
X = data[['AQI', 'BreathingRate', 'Activity']]
y = data['Filtration']

# STEP 3: Split data (training + testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 4: Create model
model = RandomForestRegressor(n_estimators=100)

# STEP 5: Train model
model.fit(X_train, y_train)

# STEP 6: Evaluate model
score = model.score(X_test, y_test)
print("Model Accuracy (R²):", score)

# STEP 7: Save model
joblib.dump(model, "filtration_model.pkl")

print("Model trained and saved successfully!")
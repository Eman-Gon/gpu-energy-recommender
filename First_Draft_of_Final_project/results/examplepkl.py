import joblib
import numpy as np

# Load model and scaler
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example: Predict for a new hour
new_hour = {
    'price_mwh': 45.0,
    'gpu_utilization_pct': 50.0,
    'active_jobs': 100,
    'power_consumption_kw': 150.0,
    'hourly_cost_usd': 6.75,
    'hour': 14,  # 2pm
    'day_of_week': 5,  # Saturday
    'is_weekend': 1,
    'is_business_hours': 0,
    'is_peak_hours': 0
}

# Convert to array (in correct order!)
X = np.array([[
    new_hour['price_mwh'],
    new_hour['gpu_utilization_pct'],
    new_hour['active_jobs'],
    new_hour['power_consumption_kw'],
    new_hour['hourly_cost_usd'],
    new_hour['hour'],
    new_hour['day_of_week'],
    new_hour['is_weekend'],
    new_hour['is_business_hours'],
    new_hour['is_peak_hours']
]])

# Predict
prediction = rf_model.predict(X)[0]
probability = rf_model.predict_proba(X)[0]

print(f"Prediction: {'Efficient' if prediction == 1 else 'Inefficient'}")
print(f"Confidence: {probability[prediction]:.1%}")
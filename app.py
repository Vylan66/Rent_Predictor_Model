import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
MODEL_FILENAME = 'rent_market_forecaster_prophet.pkl'
INCREASE_THRESHOLD_DOLLARS = 15

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Load the dictionary of Prophet models at startup ---
try:
    prophet_models = joblib.load(MODEL_FILENAME)
    print(f"✅ All {len(prophet_models)} Prophet models loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Model file '{MODEL_FILENAME}' not found. Please run 'train_model.py' first.")
    prophet_models = None

# --- Prediction Logic ---
def find_next_increase(forecast_df):
    """Analyzes a forecast to find the first significant rent increase."""
    forecast_df['increase'] = forecast_df['yhat'].diff()
    next_increase_row = forecast_df[forecast_df['increase'] >= INCREASE_THRESHOLD_DOLLARS]
    if not next_increase_row.empty:
        increase_date = next_increase_row.iloc[0]['ds']
        return increase_date.strftime('%B %Y')
    return "Not within the next 18 months"

def get_confidence_level(mape):
    """Converts a MAPE score into a simple confidence level."""
    if mape < 2.5:
        return "High"
    elif mape < 5:
        return "Medium"
    else:
        return "Low"

@app.route('/predict', methods=['POST'])
def predict():
    if not prophet_models:
        return jsonify({"error": "Models not loaded. Please check server logs."}), 500

    data = request.get_json()
    
    # --- 1. Select the correct model from our dictionary ---
    # FIX: Explicitly cast integer types from JSON to prevent potential key mismatches
    # (e.g., Python `int` vs. numpy `int64` used during model training).
    model_key = (
        data['suburb'],
        data['property_type'],
        int(data['bedrooms']),
        int(data['bathrooms'])
    )
    
    if model_key not in prophet_models:
        return jsonify({"error": f"No forecast model available for this specific property configuration in {data['suburb']}."}), 404
        
    # Access the model and its validation score from the loaded dictionary
    selected_model_data = prophet_models[model_key]
    selected_model = selected_model_data['model']
    validation_mape = selected_model_data['validation_mape']

    # --- 2. Generate a future forecast ---
    # Create a future dataframe that includes the cap and floor for prediction
    future_dates = selected_model.make_future_dataframe(periods=6, freq='QS-JAN') # 6 Quarters = 18 months
    future_dates['floor'] = selected_model.history['floor'].iloc[0]
    future_dates['cap'] = selected_model.history['cap'].iloc[0]

    forecast = selected_model.predict(future_dates)

    # --- 3. Analyze and Format Results ---
    # FIX: Make this logic more robust by filtering for future dates instead of using fixed indices.
    last_historical_date = selected_model.history['ds'].iloc[-1]
    quarterly_forecast = forecast[forecast['ds'] > last_historical_date]

    if quarterly_forecast.empty:
        return jsonify({"error": "Could not generate a future forecast."}), 500

    current_market_average = round(quarterly_forecast['yhat'].iloc[0], 2)
    
    next_increase_date_str = find_next_increase(quarterly_forecast)
    confidence = get_confidence_level(validation_mape)

    chart_data = {
        'labels': [d.strftime('%b %Y') for d in quarterly_forecast['ds']],
        'data': [round(p, 0) for p in quarterly_forecast['yhat']]
    }

    return jsonify({
        "current_market_average": current_market_average,
        "next_increase_prediction": next_increase_date_str,
        "forecast_chart_data": chart_data,
        "request_details": data,
        "confidence": confidence
    })

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)


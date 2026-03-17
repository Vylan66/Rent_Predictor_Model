import pandas as pd
from prophet import Prophet
import joblib
import warnings
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
CSV_FILENAME = 'perth_rent_history.csv'
MODEL_FILENAME = 'rent_market_forecaster_prophet.pkl'

# --- Main Model Training Function ---
def train_prophet_models():
    """Loads data, validates, then trains a separate Prophet model for each property time-series."""
    print(f"Loading historical data from '{CSV_FILENAME}'...")
    try:
        df = pd.read_csv(CSV_FILENAME)
    except FileNotFoundError:
        print(f"\n❌ ERROR: '{CSV_FILENAME}' not found. Please run 'generate_data.py' first.")
        return

    print("Data loaded. Preparing to train and validate multiple Prophet models...")
    df['date'] = pd.to_datetime(df['date'])

    # Prophet requires specific column names: 'ds' for the date and 'y' for the value.
    df.rename(columns={'date': 'ds', 'average_rent': 'y'}, inplace=True)
    
    grouping_cols = ['suburb', 'property_type', 'bedrooms', 'bathrooms']
    
    # Create a dictionary to hold all our trained models and their validation scores
    prophet_models = {}
    validation_scores = []
    
    # Group the dataframe by each unique property configuration
    grouped = df.groupby(grouping_cols)
    total_groups = len(grouped)
    print(f"Found {total_groups} unique property types to model.")

    for i, (group_name, group_df_original) in enumerate(grouped):
        # Use .copy() to avoid SettingWithCopyWarning
        group_df = group_df_original.copy()

        print(f"Processing model {i+1}/{total_groups} for: {group_name}...", end='\r')
        
        # Skip training if there's not enough data for a meaningful forecast
        if len(group_df) < 5:
            continue

        model_key = group_name

        # --- ADDING A REALISTIC FLOOR AND CAP TO THE DATA ---
        min_rent = group_df['y'].min()
        max_rent = group_df['y'].max()
        # Floor prevents negative values
        group_df['floor'] = max(250, min_rent * 0.8)
        # Cap prevents runaway positive values. Assume rent won't triple in the forecast period.
        group_df['cap'] = max_rent * 3.0

        # --- 1. VALIDATION STEP ---
        # Train on all data except the last two quarters, then test on those two.
        train_set = group_df.iloc[:-2]
        test_set = group_df.iloc[-2:]

        validation_model = Prophet(growth='logistic', seasonality_mode='additive', daily_seasonality=False, weekly_seasonality=False)
        validation_model.fit(train_set[['ds', 'y', 'floor', 'cap']])
        
        # Create a future dataframe that covers the test period
        future_validation_df = validation_model.make_future_dataframe(periods=2, freq='QS-JAN')
        # Add cap and floor to the future dataframe for prediction
        future_validation_df['floor'] = group_df['floor'].iloc[0]
        future_validation_df['cap'] = group_df['cap'].iloc[0]
        
        forecast = validation_model.predict(future_validation_df)
        
        # Get the predictions that correspond to the test set dates
        predictions = forecast.iloc[-2:]['yhat']
        actuals = test_set['y']
        
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        validation_scores.append(mape)

        # --- 2. FINAL MODEL TRAINING ---
        # Now, train the final model on ALL available data for this property type
        final_model = Prophet(growth='logistic', seasonality_mode='additive', daily_seasonality=False, weekly_seasonality=False)
        final_model.fit(group_df[['ds', 'y', 'floor', 'cap']])
        
        # Store the trained model and its validation score
        prophet_models[model_key] = {
            'model': final_model,
            'validation_mape': mape
        }

    print(f"\nModel training and validation complete. Trained {len(prophet_models)} individual models.")

    # --- Print Validation Summary ---
    avg_mape = sum(validation_scores) / len(validation_scores)
    print(f"\n--- Validation Summary ---")
    print(f"Average Model Error (MAPE): {avg_mape:.2f}%")
    print("This means, on average, a model's forecast for the last 6 months was off by this percentage.")
    print(f"--------------------------")

    # --- Save the entire dictionary of models to a single file ---
    joblib.dump(prophet_models, MODEL_FILENAME)
    
    print(f"\n✅ All Prophet models successfully saved to '{MODEL_FILENAME}'")

# --- Main Execution ---
if __name__ == "__main__":
    train_prophet_models()
 

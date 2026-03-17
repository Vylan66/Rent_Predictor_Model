import pandas as pd
import numpy as np

# --- Configuration ---
CSV_FILENAME = 'perth_rent_history.csv'
RANDOM_SEED = 42 # This ensures the "random" data is the same every time.

# Base rent for a 3-bed, 1-bath house in Jan 2022.
# Also includes growth rate and volatility for simulation.
SUBURB_PROFILES = {
    'Nedlands':      {'base': 700, 'growth': 1.065, 'volatility': 0.02},
    'Canning Vale':  {'base': 580, 'growth': 1.055, 'volatility': 0.015},
    'Joondalup':     {'base': 550, 'growth': 1.058, 'volatility': 0.018},
    'Fremantle':     {'base': 680, 'growth': 1.070, 'volatility': 0.03},
    'Perth CBD':     {'base': 750, 'growth': 1.060, 'volatility': 0.04},
    'Scarborough':   {'base': 720, 'growth': 1.075, 'volatility': 0.035},
    'Armadale':      {'base': 450, 'growth': 1.045, 'volatility': 0.01},
    'Victoria Park': {'base': 620, 'growth': 1.068, 'volatility': 0.025},
    'Cottesloe':     {'base': 950, 'growth': 1.080, 'volatility': 0.05},
    'Kalamunda':     {'base': 590, 'growth': 1.050, 'volatility': 0.01},
    'Mandurah':      {'base': 480, 'growth': 1.060, 'volatility': 0.02},
    'Rockingham':    {'base': 500, 'growth': 1.055, 'volatility': 0.015},
    'Midland':       {'base': 510, 'growth': 1.052, 'volatility': 0.012},
}

# Modifiers to calculate rent based on property features relative to the base.
PROPERTY_TYPE_MODIFIERS = {'House': 1.0, 'Townhouse': 0.95, 'Apartment': 0.85}
BEDROOMS_MODIFIERS = {1: 0.70, 2: 0.88, 3: 1.0, 4: 1.15, 5: 1.30}
BATHROOMS_MODIFIERS = {1: 1.0, 2: 1.10, 3: 1.20}

# --- Data Generation Function ---

def generate_granular_data():
    """Generates a detailed, granular dataset for multiple property types."""
    print(f"Generating granular historical market data using random seed {RANDOM_SEED}...")
    
    # Set the seed to make our 'random' numbers predictable
    np.random.seed(RANDOM_SEED)
    
    market_data = []
    date_range = pd.date_range(start='2022-01-01', end='2025-07-01', freq='QS-JAN')

    for suburb, profile in SUBURB_PROFILES.items():
        base_rent_over_time = []
        current_base = profile['base']
        
        # First, calculate the base rent trend for the suburb
        for date in date_range:
            quarterly_growth = (profile['growth'] ** (1/4))
            noise = np.random.normal(1, profile['volatility'])
            seasonality = 1
            if date.month in [7, 10]: seasonality = 0.98  # Winter/Spring dip
            if date.month in [1, 4]: seasonality = 1.02   # Summer/Autumn spike
            current_base = current_base * quarterly_growth * noise * seasonality
            base_rent_over_time.append(int(round(current_base, -1)))

        # Now, generate variations for each time point
        for i, date in enumerate(date_range):
            base_rent = base_rent_over_time[i]
            for p_type, p_mod in PROPERTY_TYPE_MODIFIERS.items():
                for beds, bed_mod in BEDROOMS_MODIFIERS.items():
                    # Apartments rarely have 5 bedrooms
                    if p_type == 'Apartment' and beds > 3:
                        continue
                    for baths, bath_mod in BATHROOMS_MODIFIERS.items():
                        # Unlikely to have more bathrooms than bedrooms
                        if baths > beds:
                            continue
                        # Unlikely to have a 1-bed, 3-bath property
                        if beds == 1 and baths > 1:
                            continue
                        
                        rent = base_rent * p_mod * bed_mod * bath_mod
                        
                        market_data.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'suburb': suburb,
                            'property_type': p_type,
                            'bedrooms': beds,
                            'bathrooms': baths,
                            'average_rent': int(round(rent, 0))
                        })

    return pd.DataFrame(market_data)

# --- Main Execution ---
if __name__ == "__main__":
    df_granular = generate_granular_data()
    df_granular.to_csv(CSV_FILENAME, index=False)
    print(f"\n✅ Granular historical data successfully saved to '{CSV_FILENAME}'")
    print(f"Total records generated: {len(df_granular)}")
    print("\nSample of the generated data:")
    print(df_granular.sample(5, random_state=RANDOM_SEED)) # Use seed on sample for consistency

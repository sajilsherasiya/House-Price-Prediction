"""
Inference Script
Demonstrates how to load the persisted pipeline and run predictions on new samples.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

# Ensure src directory is in path for custom transformers
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir / "src"))



def predict_sample(features_dict):
    """
    Takes a dictionary of house features and returns predicted house value in dollars.
    """
    model_path = Path(__file__).resolve().parent / "models" / "house_price_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")

    pipeline = joblib.load(model_path)
    df = pd.DataFrame([features_dict])
    pred_100k = pipeline.predict(df)[0]
    return pred_100k * 100_000


if __name__ == "__main__":
    # Example property in San Francisco Bay Area
    sample_house = {
        "MedInc": 8.3252,       # High median income ($83,252)
        "HouseAge": 41.0,       # 41 years old
        "AveRooms": 6.9841,     # ~7 rooms
        "AveBedrms": 1.0238,    # ~1 bedroom ratio
        "Population": 322.0,    # Low density
        "AveOccup": 2.5555,     # ~2.5 occupants
        "Latitude": 37.88,      # Bay Area latitude
        "Longitude": -122.23    # Bay Area longitude
    }

    predicted_price = predict_sample(sample_house)
    print("--- Sample House Price Prediction ---")
    for k, v in sample_house.items():
        print(f"  {k}: {v}")
    print(f"\nEstimated Valuation: ${predicted_price:,.2f} USD")

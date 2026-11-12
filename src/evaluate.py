"""
Model Evaluation Metrics & Utilities
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(name, y_true, y_pred):
    """
    Computes and prints standard regression metrics:
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - R-squared (R2)
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- Model Evaluation: {name} ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     ${rmse * 100_000:,.2f} ({rmse:.4f} in units of $100k)")
    print(f"MAE:      ${mae * 100_000:,.2f} ({mae:.4f} in units of $100k)")

    return {
        "model": name,
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "rmse_dollars": float(rmse * 100_000),
        "mae_dollars": float(mae * 100_000),
    }

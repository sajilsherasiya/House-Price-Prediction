"""
Model Training & Benchmarking Pipeline
Trains multiple regression models, compares metrics, and persists the best pipeline.
"""

import os
import json
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from data_loader import get_train_test_split
from feature_engineering import HousingFeatureAdder
from evaluate import evaluate_model


def benchmark_and_train():
    print("Step 1: Loading California Housing dataset...")
    X_train, X_test, y_train, y_test = get_train_test_split(test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Candidate models
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
        "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(max_iter=200, random_state=42),
    }

    results = []
    trained_pipelines = {}

    print("\nStep 2: Training and benchmarking candidate models...")
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("feature_adder", HousingFeatureAdder(add_geo_distances=True)),
            ("scaler", RobustScaler()),
            ("regressor", estimator)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(name, y_test, y_pred)
        results.append(metrics)
        trained_pipelines[name] = pipeline

    # Select best model based on R2 score / lowest RMSE
    best_result = max(results, key=lambda x: x["r2"])
    best_name = best_result["model"]
    best_pipeline = trained_pipelines[best_name]

    print(f"\n=======================================================")
    print(f" Best Model Selected: {best_name}")
    print(f" R² Score: {best_result['r2']:.4f}")
    print(f" RMSE: ${best_result['rmse_dollars']:,.2f}")
    print(f"=======================================================")

    # Create models directory
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = model_dir / "house_price_pipeline.joblib"
    metadata_path = model_dir / "model_metadata.json"

    print(f"\nSaving model pipeline to: {pipeline_path}")
    joblib.dump(best_pipeline, pipeline_path)

    metadata = {
        "best_model": best_name,
        "all_benchmarks": results,
        "input_features": list(X_train.columns),
        "target_description": "Median house value (in $100,000s)",
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    benchmark_and_train()

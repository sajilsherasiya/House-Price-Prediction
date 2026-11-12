"""
Data Loader Module for House Price Prediction
Loads the California Housing dataset and handles train/test splitting.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_housing_data(as_frame=True):
    """
    Fetches the California Housing dataset.
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target (Median house value in $100,000s)
    """
    california = fetch_california_housing(as_frame=as_frame)
    X = california.data
    y = california.target
    return X, y


def get_train_test_split(test_size=0.2, random_state=42):
    """
    Loads data and performs a stratified/reproducible train-test split.
    """
    X, y = load_housing_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y = load_housing_data()
    print("Dataset Shape:", X.shape)
    print("\nFeatures Overview:")
    print(X.head())
    print("\nTarget Summary:")
    print(y.describe())

"""
Feature Engineering Module
Implements custom Scikit-Learn transformers for domain-specific feature creation.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HousingFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create meaningful real-estate ratios:
    1. Bedrooms per room: AveBedrms / AveRooms
    2. Rooms per household: AveRooms
    3. Population per household: Population / (Population / AveOccup) -> AveOccup
    4. Proximity to Los Angeles (34.0522, -118.2437)
    5. Proximity to San Francisco (37.7749, -122.4194)
    """

    def __init__(self, add_geo_distances=True):
        self.add_geo_distances = add_geo_distances

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert to DataFrame if NumPy array
        if isinstance(X, np.ndarray):
            cols = [
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
            ]
            df = pd.DataFrame(X, columns=cols)
        else:
            df = X.copy()

        # Ratio features
        df["BedroomsPerRoom"] = df["AveBedrms"] / np.maximum(df["AveRooms"], 1e-5)
        df["RoomsPerPerson"] = df["AveRooms"] / np.maximum(df["AveOccup"], 1e-5)

        if self.add_geo_distances:
            # Coordinates of major economic centers
            la_coord = (34.0522, -118.2437)
            sf_coord = (37.7749, -122.4194)

            # Euclidean proxy distance
            df["DistToLA"] = np.sqrt(
                (df["Latitude"] - la_coord[0]) ** 2
                + (df["Longitude"] - la_coord[1]) ** 2
            )
            df["DistToSF"] = np.sqrt(
                (df["Latitude"] - sf_coord[0]) ** 2
                + (df["Longitude"] - sf_coord[1]) ** 2
            )
            df["MinDistToHub"] = np.minimum(df["DistToLA"], df["DistToSF"])

        return df

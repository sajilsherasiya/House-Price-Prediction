"""
Exploratory Data Analysis (EDA) Script
Generates statistical summaries and feature correlations.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_housing_data


def run_eda(save_plots=True):
    print("=== California Housing Exploratory Data Analysis ===")
    X, y = load_housing_data()
    df = X.copy()
    df["MedHouseVal"] = y

    print("\n--- Shape & Missing Values ---")
    print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing Values:\n{df.isnull().sum()}")

    print("\n--- Summary Statistics ---")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])

    # Correlation Matrix
    corr_matrix = df.corr()
    print("\n--- Correlation with Target (MedHouseVal) ---")
    print(corr_matrix["MedHouseVal"].sort_values(ascending=False))

    if save_plots:
        output_dir = Path(__file__).resolve().parent / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("California Housing Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        heatmap_file = output_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_file, dpi=150)
        plt.close()
        print(f"\nSaved correlation heatmap to: {heatmap_file}")

        # 2. Target Value Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(df["MedHouseVal"], kde=True, bins=40, color="teal")
        plt.title("Distribution of Median House Value ($100k)", fontsize=13, fontweight="bold")
        plt.xlabel("Median House Value ($100,000)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        dist_file = output_dir / "target_distribution.png"
        plt.savefig(dist_file, dpi=150)
        plt.close()
        print(f"Saved target distribution plot to: {dist_file}")


if __name__ == "__main__":
    run_eda()

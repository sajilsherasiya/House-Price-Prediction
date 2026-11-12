"""
Interactive Streamlit Web Application for House Price Prediction
"""

import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src directory is in path for custom unpickling
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir / "src"))
from feature_engineering import HousingFeatureAdder


st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_metadata():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "house_price_pipeline.joblib"
    meta_path = base_dir / "models" / "model_metadata.json"

    if not model_path.exists():
        st.error(
            "Trained model not found! Please run `python src/train.py` first to train and save the model."
        )
        st.stop()

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return model, metadata


# Load artifacts
model, metadata = load_model_and_metadata()

# Header
st.markdown(
    '<div class="main-title">🏡 California Real Estate Price Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">An end-to-end Data Science & Machine Learning application with feature engineering and ensemble modeling.</div>',
    unsafe_allow_html=True,
)

# Sidebar - Location presets & property attributes
st.sidebar.header("📍 Property Location & Presets")

presets = {
    "San Francisco (Bay Area)": (37.7749, -122.4194),
    "Silicon Valley (San Jose)": (37.3382, -121.8863),
    "Los Angeles (Westside)": (34.0522, -118.2437),
    "San Diego (Coastal)": (32.7157, -117.1611),
    "Sacramento (Central Valley)": (38.5816, -121.4944),
    "Custom Coordinates": None,
}

selected_preset = st.sidebar.selectbox("Choose City Preset", list(presets.keys()))

if selected_preset != "Custom Coordinates":
    default_lat, default_lon = presets[selected_preset]
else:
    default_lat, default_lon = 36.7783, -119.4179

latitude = st.sidebar.slider(
    "Latitude",
    min_value=32.5,
    max_value=42.0,
    value=float(default_lat),
    step=0.01,
)
longitude = st.sidebar.slider(
    "Longitude",
    min_value=-124.5,
    max_value=-114.0,
    value=float(default_lon),
    step=0.01,
)

st.sidebar.header("🏠 Neighborhood & Home Specs")

med_inc = st.sidebar.slider(
    "Median Household Income ($10,000s)",
    min_value=0.5,
    max_value=15.0,
    value=4.5,
    step=0.1,
    help="Ex: 4.5 represents $45,000 median annual household income",
)

house_age = st.sidebar.slider(
    "Median House Age (Years)", min_value=1, max_value=52, value=25, step=1
)

ave_rooms = st.sidebar.slider(
    "Average Rooms per House", min_value=1.0, max_value=12.0, value=5.4, step=0.1
)

ave_bedrms = st.sidebar.slider(
    "Average Bedrooms per House",
    min_value=0.5,
    max_value=5.0,
    value=1.1,
    step=0.1,
)

population = st.sidebar.number_input(
    "Block Group Population",
    min_value=50,
    max_value=30000,
    value=1400,
    step=50,
)

ave_occup = st.sidebar.slider(
    "Average Household Occupants",
    min_value=1.0,
    max_value=8.0,
    value=2.8,
    step=0.1,
)

# Prepare dataframe for prediction matching training schema
input_df = pd.DataFrame(
    [
        {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }
    ]
)

# Prediction
pred_value_100k = model.predict(input_df)[0]
estimated_dollars = pred_value_100k * 100_000

# Layout Main Columns
col1, col2 = st.columns([1.2, 1.8], gap="large")

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 1.1rem; opacity: 0.9;">Estimated Median House Value</div>
            <div class="metric-val">${estimated_dollars:,.0f}</div>
            <div style="font-size: 0.9rem; opacity: 0.85;">Model: {metadata.get('best_model', 'Ensemble Regressor')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Key Derived Factors")
    rooms_per_bed = ave_rooms / max(ave_bedrms, 0.01)
    la_dist = np.sqrt(
        (latitude - 34.0522) ** 2 + (longitude - (-118.2437)) ** 2
    ) * 111.0  # Approx km
    sf_dist = np.sqrt(
        (latitude - 37.7749) ** 2 + (longitude - (-122.4194)) ** 2
    ) * 111.0

    st.write(f"- **Rooms per Bedroom:** `{rooms_per_bed:.2f}`")
    st.write(f"- **Est. Distance to SF Bay:** `{sf_dist:.1f} km`")
    st.write(f"- **Est. Distance to LA Metro:** `{la_dist:.1f} km`")
    st.write(
        f"- **Median Household Income:** `${med_inc * 10_000:,.0f}/year`"
    )

    st.info(
        "💡 **Tip:** Median income and geographic proximity to coastal metro regions are the strongest drivers of housing valuation."
    )

with col2:
    st.subheader("🗺️ Geographic Position")
    # Streamlit map requires 'lat' and 'lon' or 'latitude' and 'longitude'
    map_data = pd.DataFrame(
        [
            {
                "latitude": latitude,
                "longitude": longitude,
                "label": "Selected Location",
            }
        ]
    )
    st.map(map_data, zoom=6)

# Benchmarks & Metrics Tabs
st.write("---")
tab1, tab2 = st.tabs(["📊 Model Benchmark Performance", "🔬 Data Science Workflow"])

with tab1:
    st.subheader("Candidate Model Benchmark Comparison")
    benchmark_df = pd.DataFrame(metadata.get("all_benchmarks", []))
    if not benchmark_df.empty:
        display_df = benchmark_df.copy()
        display_df["RMSE ($ USD)"] = display_df["rmse_dollars"].apply(
            lambda x: f"${x:,.2f}"
        )
        display_df["MAE ($ USD)"] = display_df["mae_dollars"].apply(
            lambda x: f"${x:,.2f}"
        )
        display_df["R² Score"] = display_df["r2"].apply(lambda x: f"{x:.4f}")
        display_df = display_df[
            ["model", "R² Score", "RMSE ($ USD)", "MAE ($ USD)"]
        ].rename(columns={"model": "Model"})
        st.dataframe(display_df, use_container_width=True)

with tab2:
    st.markdown(
        """
    ### End-to-End Pipeline Overview:
    1. **Data Acquisition**: California Housing Dataset (20,640 census blocks, 8 primary features).
    2. **Exploratory Data Analysis**: Handled geographic non-linearities, skewness, and multi-collinearity.
    3. **Feature Engineering**: Created real-estate ratios (`BedroomsPerRoom`, `RoomsPerPerson`) and geographic proximity proxies (`DistToLA`, `DistToSF`).
    4. **Transformation**: Outlier-resilient `RobustScaler` coupled into an atomic Scikit-Learn `Pipeline`.
    5. **Benchmarking & Selection**: Evaluated Linear, Ridge, Random Forest, and Gradient Boosted Regressors using 5-fold CV & holdout test evaluation.
    """
    )

"""
Data Preprocessing Pipeline Module.
Contains dataset loaders, data cleaning routines, anomaly detection,
categorical encoders, and feature scaling transformers for both ML models.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Features for Crop Recommendation
CROP_REC_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_REC_TARGET = "label"

# Features for Yield Prediction
YIELD_NUMERICAL_FEATURES = [
    "Temperature", "Rainfall", "Humidity", "pH",
    "Nitrogen", "Phosphorus", "Potassium", "Fertilizer", "Pesticide"
]
YIELD_CATEGORICAL_FEATURES = ["Crop", "Season", "Soil Type", "State"]
YIELD_TARGET = "Yield"

def get_dataset_path(filename: str) -> str:
    """Dynamically locates dataset files in datasets/ or current working directory."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "datasets", filename),
        os.path.join("datasets", filename),
        filename
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Could not find dataset file: {filename}")

# =========================================================================
# 1. CROP RECOMMENDATION PREPROCESSING
# =========================================================================

def load_and_clean_crop_rec_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and cleans Crop_recommendation.csv.
    Validates physical bounds, removes duplicate rows, and strips string whitespace.
    """
    path = file_path or get_dataset_path("Crop_recommendation.csv")
    df = pd.read_csv(path)

    # Validate required columns
    required_cols = CROP_REC_FEATURES + [CROP_REC_TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Crop recommendation dataset missing columns: {missing}")

    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = initial_len - len(df)

    # Clean target
    df[CROP_REC_TARGET] = df[CROP_REC_TARGET].astype(str).str.strip().str.lower()

    # Handle missing values
    df = df.dropna(subset=required_cols).copy()

    # Validate physical ranges
    df = df[
        (df["N"] >= 0) & (df["N"] <= 300) &
        (df["P"] >= 0) & (df["P"] <= 300) &
        (df["K"] >= 0) & (df["K"] <= 300) &
        (df["ph"] >= 1.0) & (df["ph"] <= 14.0) &
        (df["humidity"] >= 0.0) & (df["humidity"] <= 100.0) &
        (df["rainfall"] >= 0.0)
    ].copy()

    return df

def get_crop_rec_splits(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, StandardScaler]:
    """
    Prepares train/test splits and fitted preprocessor objects for Crop Recommendation.
    """
    if df is None:
        df = load_and_clean_crop_rec_data()

    X = df[CROP_REC_FEATURES].values
    y_raw = df[CROP_REC_TARGET].values

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, scaler

# =========================================================================
# 2. CROP YIELD PREPROCESSING
# =========================================================================

def extract_state_from_region(region_series: pd.Series) -> pd.Series:
    """
    Extracts and standardizes State names from 'District, State' strings in crop_yield_dataset.csv.
    """
    def clean_state(val):
        if not isinstance(val, str):
            return "All India"
        parts = val.split(",")
        state_str = parts[-1].strip()
        # Normalization of common variations
        state_map = {
            "AndhraPradesh": "Andhra Pradesh",
            "Himachal  Pradesh": "Himachal Pradesh",
            "Himachal Pradesh": "Himachal Pradesh",
            "Uttar Pradesh": "Uttar Pradesh",
            "Madhya Pradesh": "Madhya Pradesh",
            "Tamil Nadu": "Tamil Nadu",
            "West Bengal": "West Bengal"
        }
        return state_map.get(state_str, state_str)

    return region_series.apply(clean_state)

def load_and_clean_yield_data(
    file_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads and cleans crop_yield_dataset.csv.
    Extracts State, clips anomalous negative temperatures, validates Yield > 0,
    and removes duplicates.
    """
    path = file_path or get_dataset_path("crop_yield_dataset.csv")
    df = pd.read_csv(path)

    # Validate essential columns
    base_cols = ["Crop", "Season", "Soil Type", "Yield"] + YIELD_NUMERICAL_FEATURES
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Crop yield dataset missing required columns: {missing}")

    # Extract State from Region
    if "Region" in df.columns:
        df["State"] = extract_state_from_region(df["Region"])
    else:
        df["State"] = "All India"

    # Drop duplicate records
    df = df.drop_duplicates(subset=YIELD_CATEGORICAL_FEATURES + YIELD_NUMERICAL_FEATURES).copy()

    # Drop nulls in target and features
    df = df.dropna(subset=YIELD_CATEGORICAL_FEATURES + YIELD_NUMERICAL_FEATURES + [YIELD_TARGET]).copy()

    # Sanitize anomaly values:
    # 1. Yield must be strictly positive
    df = df[df[YIELD_TARGET] > 0].copy()
    # 2. Clip negative temperatures (outliers/sensor anomalies) to reasonable ambient temp (5.0 C)
    df["Temperature"] = df["Temperature"].clip(lower=5.0, upper=50.0)
    # 3. Clip pH to valid soil agronomic range [3.0, 10.0]
    df["pH"] = df["pH"].clip(lower=3.0, upper=10.0)

    # Optional downsampling if specified for rapid testing
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    return df

def build_yield_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for Yield Prediction:
    - StandardScaler for numerical environmental/nutrient features
    - OneHotEncoder for categorical features (Crop, Season, Soil Type, State)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                YIELD_NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                YIELD_CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )
    return preprocessor

def get_yield_splits(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Prepares train/test splits and fitted preprocessor for Yield Prediction.
    """
    if df is None:
        df = load_and_clean_yield_data(max_samples=max_samples, random_state=random_state)

    X_df = df[YIELD_NUMERICAL_FEATURES + YIELD_CATEGORICAL_FEATURES]
    y = df[YIELD_TARGET].values

    # Train / Test Split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state
    )

    # Fit ColumnTransformer on training partition
    preprocessor = build_yield_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor

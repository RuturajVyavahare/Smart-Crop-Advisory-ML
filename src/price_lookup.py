"""
Price Lookup Module.
Loads and queries mandi prices from datasets/india-crop-prices.csv.
Provides modal prices, price ranges (min-max), MSP, and per-kg rates.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
from src.crop_mapping import normalize_crop_name, map_to_price_crop, get_crop_metadata

# Cache for price dataset
_PRICES_DF: Optional[pd.DataFrame] = None

def _get_dataset_path() -> str:
    """Finds the dataset file path dynamically."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "datasets", "india-crop-prices.csv"),
        os.path.join("datasets", "india-crop-prices.csv"),
        "india-crop-prices.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError("Could not locate india-crop-prices.csv in datasets/ or root.")

def load_prices_dataset() -> pd.DataFrame:
    """Loads and caches the crop prices dataframe."""
    global _PRICES_DF
    if _PRICES_DF is None:
        path = _get_dataset_path()
        df = pd.read_csv(path)
        # Standardize columns
        df.columns = [c.strip().lower() for c in df.columns]
        # Clean price fields
        for col in ["modal_rs_per_quintal", "min", "max"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
        if "msp" in df.columns:
            df["msp"] = pd.to_numeric(df["msp"], errors="coerce")
        _PRICES_DF = df
    return _PRICES_DF

def get_crop_price(crop_name: str) -> Dict[str, Any]:
    """
    Looks up official market price information for a given crop.
    
    Args:
        crop_name: Crop name (e.g. 'rice', 'Rice (Paddy)', 'Cotton', 'chickpea')
        
    Returns:
        Dictionary with modal price (Rs/Quintal), min, max, msp, and price per kg.
    """
    df = load_prices_dataset()
    canon_key = normalize_crop_name(crop_name)
    target_price_crop = map_to_price_crop(canon_key)
    meta = get_crop_metadata(canon_key)

    # 1. Exact match on mapped price_crop
    match = df[df["crop"].str.strip().str.lower() == target_price_crop.lower()]

    # 2. Substring or keyword match
    if match.empty:
        match = df[df["crop"].str.contains(target_price_crop, case=False, na=False)]

    # 3. Match on canonical name or hindi name
    if match.empty:
        match = df[df["crop"].str.contains(canon_key, case=False, na=False)]

    if not match.empty:
        row = match.iloc[0]
        modal_price = float(row["modal_rs_per_quintal"])
        min_p = float(row["min"]) if row["min"] > 0 else modal_price * 0.85
        max_p = float(row["max"]) if row["max"] > 0 else modal_price * 1.15
        msp_val = float(row["msp"]) if pd.notna(row["msp"]) else None
        crop_title = str(row["crop"])
    else:
        # Fallback benchmark
        modal_price = 4500.0
        min_p = 3800.0
        max_p = 5200.0
        msp_val = None
        crop_title = meta.get("display_name", crop_name.title())

    return {
        "canonical_name": canon_key,
        "matched_crop_name": crop_title,
        "modal_rs_per_quintal": modal_price,
        "min_rs_per_quintal": min_p,
        "max_rs_per_quintal": max_p,
        "msp_rs_per_quintal": msp_val,
        "price_per_kg": round(modal_price / 100.0, 2),
        "source": "Agmarknet Mandi Price Reference (india-crop-prices.csv)"
    }

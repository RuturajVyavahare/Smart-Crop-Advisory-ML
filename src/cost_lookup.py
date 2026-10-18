"""
Cost of Cultivation Lookup Module.
Queries CACP Comprehensive C2 and operational cultivation costs from datasets/cost-of-cultivation.csv.
Supports state-level breakdown and national agricultural benchmarks.
"""

import os
from typing import Dict, Any, Optional, List
import pandas as pd
from src.crop_mapping import normalize_crop_name, map_to_cost_crop, get_crop_metadata

_COST_DF: Optional[pd.DataFrame] = None

# Conversion constant: 1 Hectare = 2.47105 Acres
HA_TO_ACRE = 2.47105

def _get_dataset_path() -> str:
    """Finds the cost-of-cultivation.csv path dynamically."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "datasets", "cost-of-cultivation.csv"),
        os.path.join("datasets", "cost-of-cultivation.csv"),
        "cost-of-cultivation.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError("Could not locate cost-of-cultivation.csv in datasets/ or root.")

def load_cost_dataset() -> pd.DataFrame:
    """Loads and caches the cost of cultivation dataset."""
    global _COST_DF
    if _COST_DF is None:
        path = _get_dataset_path()
        df = pd.read_csv(path)
        # Clean column names
        df.columns = [c.strip().lower() for c in df.columns]
        # Clean numeric fields
        num_cols = ["cul_cost_c2", "prod_cost_c2", "opr_cost_seed", "opr_cost_fertilizer",
                    "opr_cost_hmn_lab_casual", "opr_cost_mch_lab_hired", "opr_cost", "derived_yield"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
        _COST_DF = df
    return _COST_DF

def get_available_states() -> List[str]:
    """Returns a sorted list of unique states in the cost dataset."""
    df = load_cost_dataset()
    if "state_name" in df.columns:
        return sorted([str(s).strip() for s in df["state_name"].dropna().unique() if str(s).strip()])
    return []

def get_cultivation_cost(crop_name: str, state_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Looks up C2 Cost of Cultivation (Rs/Hectare and Rs/Acre) for a crop and optional state.
    
    Args:
        crop_name: Crop name (e.g. 'Rice', 'Wheat', 'Gram', 'Banana')
        state_name: Optional Indian State name
        
    Returns:
        Dictionary containing cost metrics per hectare, per acre, production cost per quintal,
        and input cost breakdown.
    """
    df = load_cost_dataset()
    canon_key = normalize_crop_name(crop_name)
    target_cost_crop = map_to_cost_crop(canon_key)
    meta = get_crop_metadata(canon_key)

    matched_records = pd.DataFrame()
    applied_state = "National Benchmark"

    # If crop is found in CACP field crops dataset
    if target_cost_crop:
        crop_mask = df["crop_name"].str.contains(target_cost_crop, case=False, na=False)
        matched_records = df[crop_mask]

    if not matched_records.empty:
        # Check if state specific record exists
        if state_name:
            clean_state = state_name.strip().lower()
            state_match = matched_records[matched_records["state_name"].str.strip().str.lower() == clean_state]
            if not state_match.empty:
                matched_records = state_match
                applied_state = state_match.iloc[0]["state_name"]

        # Aggregate values
        c2_cost_ha = float(matched_records["cul_cost_c2"].median())
        prod_cost_q = float(matched_records["prod_cost_c2"].median())
        seed_cost = float(matched_records.get("opr_cost_seed", pd.Series([0])).median())
        fert_cost = float(matched_records.get("opr_cost_fertilizer", pd.Series([0])).median())
        lab_cost = float(matched_records.get("opr_cost_hmn_lab_casual", pd.Series([0])).median())
        mech_cost = float(matched_records.get("opr_cost_mch_lab_hired", pd.Series([0])).median())
        crop_label = target_cost_crop
        source_note = "CACP Comprehensive Survey (cost-of-cultivation.csv)"
    else:
        # Horticultural fallback
        c2_cost_ha = float(meta.get("horticultural_cost_rs_ha", 55000.0))
        prod_cost_q = round(c2_cost_ha / 50.0, 2)
        seed_cost = round(c2_cost_ha * 0.15, 2)
        fert_cost = round(c2_cost_ha * 0.25, 2)
        lab_cost = round(c2_cost_ha * 0.35, 2)
        mech_cost = round(c2_cost_ha * 0.15, 2)
        crop_label = meta.get("display_name", crop_name.title())
        source_note = "NHB / Horticultural Economic Guidelines"

    c2_cost_acre = round(c2_cost_ha / HA_TO_ACRE, 2)

    return {
        "canonical_name": canon_key,
        "matched_cost_crop": crop_label,
        "applied_state": applied_state,
        "cul_cost_c2_per_ha": round(c2_cost_ha, 2),
        "cul_cost_c2_per_acre": c2_cost_acre,
        "prod_cost_c2_per_quintal": round(prod_cost_q, 2),
        "seed_cost_per_ha": round(seed_cost, 2),
        "fertilizer_cost_per_ha": round(fert_cost, 2),
        "labor_cost_per_ha": round(lab_cost, 2),
        "machinery_cost_per_ha": round(mech_cost, 2),
        "source": source_note
    }

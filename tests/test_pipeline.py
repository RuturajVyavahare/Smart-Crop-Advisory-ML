"""
Comprehensive Test Suite for Smart Crop Advisory System.
Validates:
1. Crop name normalization and cross-dataset mapping
2. Mandi price lookup and fallback handling
3. Cost of cultivation retrieval and state breakdown
4. Profit and financial calculations, unit conversions, INR formatting
5. Crop Recommendation inference and top-k ranking
6. Yield Prediction inference and unit conversions
7. Integrated End-to-End advisory workflow
"""

import pytest
import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.crop_mapping import (
    normalize_crop_name, get_crop_metadata, map_rec_crop_to_yield_crop,
    map_to_price_crop, map_to_cost_crop, get_all_supported_crops
)
from src.price_lookup import get_crop_price
from src.cost_lookup import get_cultivation_cost, get_available_states
from src.profit_calculation import (
    convert_area_to_ha, convert_area_to_acres, format_inr, format_inr_short,
    calculate_farm_financials, HA_TO_ACRES
)
from src.crop_recommendation import CropRecommender
from src.yield_prediction import YieldPredictor

# =====================================================================
# 1. CROP MAPPING & NORMALIZATION TESTS
# =====================================================================

def test_crop_normalization():
    assert normalize_crop_name("Rice") == "rice"
    assert normalize_crop_name("  PADDY  ") == "rice"
    assert normalize_crop_name("chana") == "chickpea"
    assert normalize_crop_name("Gram (Chana)") == "chickpea"
    assert normalize_crop_name("Pigeon Peas") == "pigeonpeas"
    assert normalize_crop_name("tur") == "pigeonpeas"
    assert normalize_crop_name("arhar") == "pigeonpeas"
    assert normalize_crop_name("cotton") == "cotton"
    assert normalize_crop_name("Kapas") == "cotton"
    assert normalize_crop_name("Moong") == "mungbean"
    assert normalize_crop_name("Urad") == "blackgram"
    assert normalize_crop_name("Corn") == "maize"

def test_all_supported_crops_count():
    crops = get_all_supported_crops()
    assert len(crops) == 22
    for c in crops:
        assert "canonical_name" in c
        assert "display_name" in c
        assert "category" in c
        assert "default_season" in c

def test_cross_dataset_mappings():
    assert map_rec_crop_to_yield_crop("rice") == "Rice"
    assert map_rec_crop_to_yield_crop("maize") == "Maize"
    assert map_rec_crop_to_yield_crop("chickpea") == "Chickpea"
    assert map_rec_crop_to_yield_crop("blackgram") == "Minor Pulses"
    assert map_rec_crop_to_yield_crop("cotton") == "Cotton"

    assert map_to_price_crop("rice") == "Rice (Paddy)"
    assert map_to_price_crop("wheat") == "Wheat"
    assert map_to_price_crop("maize") == "Maize (Corn)"
    assert map_to_price_crop("chickpea") == "Gram (Chana)"

    assert map_to_cost_crop("rice") == "Paddy"
    assert map_to_cost_crop("chickpea") == "Gram"
    assert map_to_cost_crop("cotton") == "Cotton"

# =====================================================================
# 2. PRICE LOOKUP TESTS
# =====================================================================

def test_price_lookup_wheat():
    price_info = get_crop_price("Wheat")
    assert price_info["modal_rs_per_quintal"] > 0
    assert price_info["price_per_kg"] == round(price_info["modal_rs_per_quintal"] / 100.0, 2)
    assert price_info["min_rs_per_quintal"] <= price_info["max_rs_per_quintal"]

def test_price_lookup_rice():
    price_info = get_crop_price("rice")
    assert price_info["modal_rs_per_quintal"] > 0
    assert "Rice" in price_info["matched_crop_name"]

def test_price_lookup_fallback():
    price_info = get_crop_price("UnknownCropX")
    assert price_info["modal_rs_per_quintal"] > 0
    assert price_info["price_per_kg"] > 0

# =====================================================================
# 3. COST LOOKUP TESTS
# =====================================================================

def test_cost_lookup_paddy():
    cost_info = get_cultivation_cost("rice")
    assert cost_info["cul_cost_c2_per_ha"] > 10000.0
    assert cost_info["cul_cost_c2_per_acre"] > 0
    assert cost_info["prod_cost_c2_per_quintal"] > 0

def test_cost_lookup_state_specific():
    cost_info = get_cultivation_cost("cotton", state_name="Gujarat")
    assert cost_info["cul_cost_c2_per_ha"] > 0
    assert "Gujarat" in cost_info["applied_state"] or "National" in cost_info["applied_state"]

def test_available_states_list():
    states = get_available_states()
    assert len(states) >= 15
    assert "Maharashtra" in states or "Punjab" in states or "Gujarat" in states

# =====================================================================
# 4. PROFIT & FINANCIAL CALCULATIONS TESTS
# =====================================================================

def test_unit_conversions():
    # 1 ha = 2.47105 acres
    assert round(convert_area_to_ha(2.47105, "acre"), 2) == 1.0
    assert round(convert_area_to_acres(1.0, "ha"), 2) == round(HA_TO_ACRES, 2)

def test_inr_formatting():
    assert format_inr(1000) == "₹1,000"
    assert format_inr(1000, include_paise=True) == "₹1,000.00"
    assert format_inr(125000) == "₹1,25,000"
    assert format_inr(15000000) == "₹1,50,00,000"
    assert format_inr(-5000) == "-₹5,000"
    assert "Lakh" in format_inr_short(250000)
    assert "Cr" in format_inr_short(15000000)

def test_farm_financials_math():
    # 1 hectare, 2000 kg/ha yield (20 quintals), Rs 3000/quintal price, Rs 40000/ha cost
    fin = calculate_farm_financials(
        yield_kg_per_ha=2000.0,
        farm_area=1.0,
        area_unit="hectare",
        price_per_quintal=3000.0,
        cost_per_ha=40000.0
    )
    assert fin["total_production_quintals"] == 20.0
    assert fin["total_revenue"] == 60000.0
    assert fin["total_cost"] == 40000.0
    assert fin["net_profit"] == 20000.0
    assert fin["roi_percent"] == 50.0
    assert fin["breakeven_yield_kg_per_ha"] == round(40000.0 / 30.0, 2)
    assert fin["breakeven_price_per_quintal"] == 2000.0

# =====================================================================
# 5. MODEL INFERENCE & PIPELINE INTEGRATION TESTS
# =====================================================================

def test_crop_recommender_prediction():
    recommender = CropRecommender()
    out = recommender.predict(
        n=90.0, p=42.0, k=43.0,
        temperature=21.0, humidity=82.0, ph=6.5, rainfall=202.0
    )
    assert "primary_crop" in out
    assert "display_name" in out
    assert out["confidence_percent"] > 0
    assert len(out["recommendations"]) >= 1

def test_yield_predictor_prediction():
    predictor = YieldPredictor()
    out = predictor.predict(
        crop="rice", season="Kharif", soil_type="Vertisols", state="Maharashtra",
        temperature=25.0, rainfall=900.0, humidity=70.0, ph=6.5,
        nitrogen=90.0, phosphorus=40.0, potassium=40.0, fertilizer=100.0, pesticide=0.35
    )
    assert out["predicted_yield_kg_ha"] > 100.0
    assert out["predicted_yield_quintals_ha"] == round(out["predicted_yield_kg_ha"] / 100.0, 2)
    assert out["predicted_yield_kg_acre"] > 0

def test_complete_end_to_end_flow():
    # 1. Recommend
    recommender = CropRecommender()
    rec = recommender.predict(n=80, p=40, k=40, temperature=25, humidity=80, ph=6.5, rainfall=180)
    top_crop = rec["primary_crop"]
    assert top_crop is not None

    # 2. Predict Yield
    predictor = YieldPredictor()
    yd = predictor.predict(crop=top_crop, season="Kharif", state="Maharashtra")
    assert yd["predicted_yield_kg_ha"] > 0

    # 3. Lookup Price & Cost
    price = get_crop_price(top_crop)
    cost = get_cultivation_cost(top_crop, state_name="Maharashtra")
    assert price["modal_rs_per_quintal"] > 0
    assert cost["cul_cost_c2_per_ha"] > 0

    # 4. Financials
    fin = calculate_farm_financials(
        yield_kg_per_ha=yd["predicted_yield_kg_ha"],
        farm_area=3.0,
        area_unit="acre",
        price_per_quintal=price["modal_rs_per_quintal"],
        cost_per_ha=cost["cul_cost_c2_per_ha"]
    )
    assert fin["total_revenue"] > 0
    assert fin["total_cost"] > 0
    assert fin["formatted_profit"] is not None

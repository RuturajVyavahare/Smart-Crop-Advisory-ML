"""
Smart Crop Advisory System — Streamlit Web Application.
Machine Learning Based Crop Recommendation, Yield Prediction & Profit Estimation.
Integrates Random Forest & XGBoost ML pipelines, CACP Cultivation Costs,
and Mandi Market Prices into a unified agronomic decision engine.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.crop_recommendation import CropRecommender
from src.yield_prediction import YieldPredictor
from src.price_lookup import get_crop_price, load_prices_dataset
from src.cost_lookup import get_cultivation_cost, get_available_states, load_cost_dataset
from src.profit_calculation import (
    calculate_farm_financials, format_inr, format_inr_short,
    convert_area_to_ha, convert_area_to_acres
)
from src.crop_mapping import get_crop_metadata, get_all_supported_crops, normalize_crop_name

# Page Configuration
st.set_page_config(
    page_title="Smart Crop Advisory System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Top title styling */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    /* Metric Card Styling - Adaptive for Light and Dark Modes */
    .metric-card {
        background: rgba(128, 128, 128, 0.07);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border: 1px solid rgba(128, 128, 128, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(0,0,0,0.12);
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }
    .profit-positive {
        color: #22c55e !important;
    }
    .profit-negative {
        color: #ef4444 !important;
    }
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Custom Badges */
    .badge-primary {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-secondary {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        color: #92400e;
        font-size: 0.88rem;
        margin-top: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED MODEL & DATA LOADERS
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_recommender():
    return CropRecommender()

@st.cache_resource(show_spinner=False)
def load_predictor():
    return YieldPredictor()

@st.cache_data(show_spinner=False)
def get_states_cached():
    return get_available_states()

@st.cache_data(show_spinner=False)
def load_metrics_cached():
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# Load models and resources
try:
    crop_recommender = load_recommender()
    yield_predictor = load_predictor()
    available_states = get_states_cached()
    training_metrics = load_metrics_cached()
except Exception as e:
    st.error(f"Error loading trained models: {e}. Please ensure you ran 'python train_models.py' first.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR: PARAMETERS & INPUT CONFIGURATION
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/wheat.png", width=64)
st.sidebar.markdown("### 🌾 Farm & Environment Setup")

# Preset Selector
preset = st.sidebar.selectbox(
    "⚡ Select Agro-Climatic Preset:",
    [
        "Custom Parameters",
        "🌾 Gangetic Plains (Alluvial / Rice-Wheat)",
        "🌿 Deccan Plateau (Black Soil / Cotton-Gram)",
        "☀️ Semi-Arid Rajasthan (Bajra / Pulses)",
        "🌴 Coastal Humid (Paddy / Plantation)"
    ]
)

# Preset Values Configuration
defaults = {
    "n": 90.0, "p": 42.0, "k": 43.0,
    "temp": 24.5, "hum": 75.0, "ph": 6.8, "rain": 180.0,
    "state": "Uttar Pradesh", "soil": "Inceptisols", "season": "Kharif",
    "area": 2.5, "unit": "acre"
}

if "Gangetic Plains" in preset:
    defaults.update({
        "n": 95.0, "p": 48.0, "k": 45.0, "temp": 24.0, "hum": 80.0, "ph": 7.0, "rain": 200.0,
        "state": "Uttar Pradesh", "soil": "Inceptisols", "season": "Kharif"
    })
elif "Deccan Plateau" in preset:
    defaults.update({
        "n": 115.0, "p": 42.0, "k": 22.0, "temp": 26.5, "hum": 68.0, "ph": 7.6, "rain": 85.0,
        "state": "Maharashtra", "soil": "Vertisols", "season": "Kharif"
    })
elif "Semi-Arid" in preset:
    defaults.update({
        "n": 40.0, "p": 25.0, "k": 20.0, "temp": 30.0, "hum": 45.0, "ph": 8.0, "rain": 45.0,
        "state": "Rajasthan", "soil": "Aridisols", "season": "Kharif"
    })
elif "Coastal Humid" in preset:
    defaults.update({
        "n": 80.0, "p": 55.0, "k": 45.0, "temp": 28.0, "hum": 88.0, "ph": 6.2, "rain": 240.0,
        "state": "Kerala", "soil": "Alfisols", "season": "Whole Year"
    })

# Collapsible Sidebar Sections
with st.sidebar.expander("🧪 Soil Nutrients (NPK & pH)", expanded=True):
    col_n, col_p = st.columns(2)
    with col_n:
        input_n = st.number_input("Nitrogen (N) kg/ha", min_value=0.0, max_value=250.0, value=defaults["n"], step=1.0)
    with col_p:
        input_p = st.number_input("Phosphorus (P) kg/ha", min_value=0.0, max_value=200.0, value=defaults["p"], step=1.0)

    col_k, col_ph = st.columns(2)
    with col_k:
        input_k = st.number_input("Potassium (K) kg/ha", min_value=0.0, max_value=250.0, value=defaults["k"], step=1.0)
    with col_ph:
        input_ph = st.number_input("Soil pH", min_value=3.5, max_value=10.0, value=defaults["ph"], step=0.1)

with st.sidebar.expander("🌦️ Climate & Weather Parameters", expanded=True):
    input_temp = st.slider("Temperature (°C)", min_value=8.0, max_value=48.0, value=defaults["temp"], step=0.5)
    input_hum = st.slider("Humidity (%)", min_value=10.0, max_value=100.0, value=defaults["hum"], step=1.0)
    input_rain = st.slider("Rainfall (mm)", min_value=10.0, max_value=500.0, value=defaults["rain"], step=5.0)

with st.sidebar.expander("📍 Geographic & Agronomic Profile", expanded=True):
    state_index = available_states.index(defaults["state"]) if defaults["state"] in available_states else 0
    input_state = st.selectbox("State:", available_states, index=state_index)

    soil_types = ["Vertisols (Black)", "Inceptisols (Alluvial)", "Alfisols (Red)", "Aridisols (Sandy/Desert)", "Mollisols (Hilly/Loam)", "Entisols"]
    default_soil_idx = 1 if "Inceptisols" in defaults["soil"] else 0
    input_soil = st.selectbox("Soil Classification:", soil_types, index=default_soil_idx).split(" ")[0]

    seasons = ["Kharif (Monsoon)", "Rabi (Winter)", "Whole Year / Zaid (Summer)"]
    default_season_idx = 0 if "Kharif" in defaults["season"] else 1
    input_season = st.selectbox("Cropping Season:", seasons, index=default_season_idx).split(" ")[0]

with st.sidebar.expander("📐 Landholding & Farm Area", expanded=True):
    col_a1, col_a2 = st.columns([2, 1.2])
    with col_a1:
        farm_area_val = st.number_input("Farm Size:", min_value=0.1, max_value=500.0, value=defaults["area"], step=0.5)
    with col_a2:
        farm_area_unit = st.selectbox("Unit:", ["Acre", "Hectare"], index=0).lower()

st.sidebar.markdown("---")
st.sidebar.caption("Smart Crop Advisory System | Powered by Deep Learning & Agronomic Economics")

# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------
st.markdown('<div class="hero-title">🌾 Smart Crop Advisory System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Machine Learning Based Crop Recommendation, Yield Prediction & Profit Estimation</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# RUN MODEL PREDICTION PIPELINE
# ---------------------------------------------------------
# 1. Crop Recommendation Model
rec_result = crop_recommender.predict(
    n=input_n, p=input_p, k=input_k,
    temperature=input_temp, humidity=input_hum, ph=input_ph, rainfall=input_rain,
    top_k=3
)
recommended_crop = rec_result["primary_crop"]
crop_meta = get_crop_metadata(recommended_crop)

# 2. Yield Prediction Model
yield_result = yield_predictor.predict(
    crop=recommended_crop,
    season=input_season,
    soil_type=input_soil,
    state=input_state,
    temperature=input_temp,
    rainfall=input_rain,
    humidity=input_hum,
    ph=input_ph,
    nitrogen=input_n,
    phosphorus=input_p,
    potassium=input_k,
    fertilizer=float(crop_meta.get("default_fertilizer", 75.0)),
    pesticide=float(crop_meta.get("default_pesticide", 0.35))
)
predicted_yield_ha = yield_result["predicted_yield_kg_ha"]

# 3. Market Price Lookup
price_data = get_crop_price(recommended_crop)
modal_price = price_data["modal_rs_per_quintal"]

# 4. Cultivation Cost Lookup
cost_data = get_cultivation_cost(recommended_crop, state_name=input_state)
c2_cost_ha = cost_data["cul_cost_c2_per_ha"]

# 5. Farm Financials Calculation
financials = calculate_farm_financials(
    yield_kg_per_ha=predicted_yield_ha,
    farm_area=farm_area_val,
    area_unit=farm_area_unit,
    price_per_quintal=modal_price,
    cost_per_ha=c2_cost_ha
)

# ---------------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------------
tab_dashboard, tab_technical, tab_price_explorer = st.tabs([
    "🌾 Advisory Dashboard", "📊 Model Metrics & Visualizations", "💰 Price & Cost Reference"
])

# =========================================================================
# TAB 1: ADVISORY DASHBOARD
# =========================================================================
with tab_dashboard:
    # 6 PRIMARY METRIC CARDS
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🌱 Recommended Crop</div>
            <div class="metric-value" style="font-size: 1.35rem; color: #15803d;">{rec_result['display_name']}</div>
            <div class="metric-sub">Confidence: <b>{rec_result['confidence_percent']:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Predicted Yield</div>
            <div class="metric-value">{financials['yield_quintals_per_acre']:.1f} <span style="font-size:0.95rem; font-weight:500;">Q/Acre</span></div>
            <div class="metric-sub">{financials['yield_kg_per_ha']:,.0f} kg/ha ({financials['yield_quintals_per_ha']:.1f} Q/ha)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        msp_tag = f" (MSP ₹{price_data['msp_rs_per_quintal']:,.0f})" if price_data['msp_rs_per_quintal'] else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Market Price</div>
            <div class="metric-value">{format_inr(price_data['modal_rs_per_quintal'])}</div>
            <div class="metric-sub">Per Quintal (₹{price_data['price_per_kg']:.1f}/kg){msp_tag}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💵 Cultivation Cost</div>
            <div class="metric-value">{format_inr(cost_data['cul_cost_c2_per_acre'])}</div>
            <div class="metric-sub">Per Acre ({format_inr(cost_data['cul_cost_c2_per_ha'])}/ha)</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📈 Total Revenue</div>
            <div class="metric-value">{financials['formatted_revenue_short']}</div>
            <div class="metric-sub">Production: {financials['total_production_quintals']} Quintals</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        profit_class = "profit-positive" if financials['net_profit'] >= 0 else "profit-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🟢 Estimated Profit</div>
            <div class="metric-value {profit_class}">{financials['formatted_profit_short']}</div>
            <div class="metric-sub">ROI: <b>{financials['roi_percent']:.1f}%</b> | {format_inr(financials['profit_per_acre'])}/Acre</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    dash_col1, dash_col2 = st.columns([1.1, 1])

    with dash_col1:
        with st.container(border=True):
            st.markdown("### 🌱 Crop Recommendation Analysis")
            st.write(f"**Primary Agronomic Match:** {rec_result['display_name']} ({rec_result['category']})")
            st.info(f"💡 **Description:** {crop_meta.get('description', '')}")

            st.markdown("##### 🏆 Top Candidate Crops by Model Confidence:")
            recs = rec_result["recommendations"]
            rec_df = pd.DataFrame([
                {
                    "Candidate Crop": r["display_name"],
                    "Category": r["category"],
                    "Confidence Score (%)": r["confidence_score"]
                }
                for r in recs
            ])
            st.dataframe(rec_df, hide_index=True, use_container_width=True)

            # Soil nutrient advisory
            st.markdown("##### 🧪 Agronomic Soil Diagnosis:")
            if rec_result["soil_advisory_notes"]:
                for note in rec_result["soil_advisory_notes"]:
                    st.warning(f"⚠️ {note}")
            else:
                st.success("✅ Soil NPK nutrients and pH are within optimal agronomic thresholds for this crop.")

    with dash_col2:
        with st.container(border=True):
            st.markdown("### 📊 Financial & Production Breakdown")

            fin_table_data = [
                ("Farm Operational Size", f"{financials['input_area']} {financials['input_unit'].title()} ({financials['area_hectares']} ha)"),
                ("Total Expected Harvest", f"{financials['total_production_quintals']:,.1f} Quintals ({financials['total_production_kg']:,.0f} kg)"),
                ("Gross Revenue (Market Sale)", f"{financials['formatted_revenue']}"),
                ("Comprehensive Cultivation Cost (C2)", f"{financials['formatted_cost']}"),
                ("Estimated Net Profit", f"{financials['formatted_profit']}"),
                ("Net Profit Per Acre", f"{financials['formatted_profit_per_acre']}"),
                ("Return on Investment (ROI)", f"{financials['roi_percent']:.1f} %"),
                ("Break-Even Harvest Yield", f"{financials['breakeven_yield_kg_per_ha']:,.0f} kg/ha"),
                ("Break-Even Market Price", f"{format_inr(financials['breakeven_price_per_quintal'])} / Quintal")
            ]
            fin_df = pd.DataFrame(fin_table_data, columns=["Parameter", "Estimated Value"])
            st.dataframe(fin_df, hide_index=True, use_container_width=True)

            # Input Cost Composition
            st.markdown("##### 💵 C2 Cost Composition per Hectare:")
            cost_breakdown = pd.DataFrame({
                "Input Category": ["Seed & Planting", "Fertilizers & Nutrients", "Human Labor", "Machinery & Fuel", "Land Rent & Fixed Capital"],
                "Amount (₹/ha)": [
                    cost_data["seed_cost_per_ha"],
                    cost_data["fertilizer_cost_per_ha"],
                    cost_data["labor_cost_per_ha"],
                    cost_data["machinery_cost_per_ha"],
                    max(0.0, cost_data["cul_cost_c2_per_ha"] - (cost_data["seed_cost_per_ha"] + cost_data["fertilizer_cost_per_ha"] + cost_data["labor_cost_per_ha"] + cost_data["machinery_cost_per_ha"]))
                ]
            })
            st.bar_chart(cost_breakdown.set_index("Input Category"))

    # SENSITIVITY & SCENARIO ANALYSIS
    with st.expander("📈 Dynamic Price & Yield Scenario Simulation", expanded=False):
        st.markdown("Simulate how market price fluctuations or weather variability impact your bottom line:")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_price_delta = st.slider("Market Price Change (%)", min_value=-40, max_value=60, value=0, step=5)
        with sim_col2:
            sim_yield_delta = st.slider("Harvest Yield Change (%)", min_value=-40, max_value=60, value=0, step=5)

        adjusted_price = modal_price * (1.0 + sim_price_delta / 100.0)
        adjusted_yield = predicted_yield_ha * (1.0 + sim_yield_delta / 100.0)
        sim_financials = calculate_farm_financials(
            yield_kg_per_ha=adjusted_yield,
            farm_area=farm_area_val,
            area_unit=farm_area_unit,
            price_per_quintal=adjusted_price,
            cost_per_ha=c2_cost_ha
        )

        sim_c1, sim_c2, sim_c3 = st.columns(3)
        sim_c1.metric("Simulated Revenue", sim_financials["formatted_revenue"], f"{sim_price_delta + sim_yield_delta}%")
        sim_c2.metric("Simulated Net Profit", sim_financials["formatted_profit"], f"{sim_financials['roi_percent']:.1f}% ROI")
        sim_c3.metric("Profit Per Acre", sim_financials["formatted_profit_per_acre"])

    # Download summary report
    st.markdown("<br>", unsafe_allow_html=True)
    report_dict = {
        "Recommended Crop": [rec_result["display_name"]],
        "Cropping Season": [input_season],
        "Soil Classification": [input_soil],
        "State": [input_state],
        "Farm Area": [f"{farm_area_val} {farm_area_unit}"],
        "Predicted Yield (Q/Acre)": [financials["yield_quintals_per_acre"]],
        "Total Production (Q)": [financials["total_production_quintals"]],
        "Market Price (Rs/Q)": [modal_price],
        "Cost of Cultivation (Rs/Acre)": [cost_data["cul_cost_c2_per_acre"]],
        "Estimated Revenue (Rs)": [financials["total_revenue"]],
        "Total Cost (Rs)": [financials["total_cost"]],
        "Net Profit (Rs)": [financials["net_profit"]],
        "ROI (%)": [financials["roi_percent"]]
    }
    report_csv = pd.DataFrame(report_dict).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Advisory Report as CSV",
        data=report_csv,
        file_name=f"crop_advisory_{recommended_crop}_{input_state}.csv",
        mime="text/csv"
    )

    # DISCLAIMER BOX
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>⚠️ Official Disclaimer:</b> Profit and revenue values are estimates based on the selected reference price and cultivation-cost data and may vary with actual market and farming conditions.
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================================
# TAB 2: TECHNICAL & MODEL EVALUATION
# =========================================================================
with tab_technical:
    st.markdown("### 🔬 Machine Learning Model Evaluation & Visualizations")
    st.markdown("Both tasks evaluate **Random Forest** and **XGBoost** algorithms with reproducible seeds and automated winning model selection.")

    # Metric summary cards
    if training_metrics:
        crm = training_metrics.get("crop_recommendation", {})
        ypm = training_metrics.get("yield_prediction", {})

        mcol1, mcol2 = st.columns(2)
        with mcol1:
            with st.container(border=True):
                st.markdown("#### 🌱 1. Crop Recommendation Model")
                st.markdown(f"**Winning Model:** `{crm.get('best_model')}`")
                st.write(f"- **Test Accuracy:** `{crm.get('accuracy')}%`")
                st.write(f"- **Weighted F1-Score:** `{crm.get('f1_score')}`")
                st.write(f"- **RF Accuracy:** `{crm.get('rf_accuracy')}%` | **XGB Accuracy:** `{crm.get('xgb_accuracy')}%`")

        with mcol2:
            with st.container(border=True):
                st.markdown("#### 📊 2. Yield Prediction Regressor")
                st.markdown(f"**Winning Model:** `{ypm.get('best_model')}`")
                st.write(f"- **Coefficient of Determination (R²):** `{ypm.get('r2_score')}`")
                st.write(f"- **Root Mean Squared Error (RMSE):** `{ypm.get('rmse')} kg/ha`")
                st.write(f"- **Mean Absolute Error (MAE):** `{ypm.get('mae')} kg/ha`")
                st.write(f"- **RF R²:** `{ypm.get('rf_r2')}` | **XGB R²:** `{ypm.get('xgb_r2')}`")

    # EVALUATION PLOTS
    st.markdown("#### 📈 Saved Model Evaluation Plots")
    pcol1, pcol2 = st.columns(2)

    cm_img_path = os.path.join(os.path.dirname(__file__), "models", "crop_rec_confusion_matrix.png")
    rec_feat_path = os.path.join(os.path.dirname(__file__), "models", "crop_rec_feature_importance.png")
    yield_feat_path = os.path.join(os.path.dirname(__file__), "models", "yield_feature_importance.png")
    act_pred_path = os.path.join(os.path.dirname(__file__), "models", "yield_actual_vs_predicted.png")

    with pcol1:
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, caption="Crop Recommendation — Multi-Class Confusion Matrix", use_container_width=True)
        if os.path.exists(rec_feat_path):
            st.image(rec_feat_path, caption="Crop Recommendation — Relative Feature Importance", use_container_width=True)

    with pcol2:
        if os.path.exists(act_pred_path):
            st.image(act_pred_path, caption="Yield Prediction — Actual vs Predicted Yield (kg/ha)", use_container_width=True)
        if os.path.exists(yield_feat_path):
            st.image(yield_feat_path, caption="Yield Prediction — Top 15 Predictor Importance", use_container_width=True)

# =========================================================================
# TAB 3: PRICE & COST EXPLORER
# =========================================================================
with tab_price_explorer:
    st.markdown("### 💰 Mandi Market Price & CACP Cultivation Cost Explorer")
    st.markdown("Inspect reference datasets covering **122 market prices** and **CACP field crop cultivation cost surveys**.")

    exp_tab1, exp_tab2 = st.tabs(["Mandi Market Prices", "Cost of Cultivation"])

    with exp_tab1:
        prices_df = load_prices_dataset()
        search_crop = st.text_input("🔍 Filter Market Prices by Crop Name:", "")
        if search_crop:
            filtered_prices = prices_df[prices_df["crop"].str.contains(search_crop, case=False, na=False)]
        else:
            filtered_prices = prices_df

        st.dataframe(
            filtered_prices[["crop", "modal_rs_per_quintal", "min", "max", "msp"]].rename(
                columns={
                    "crop": "Crop",
                    "modal_rs_per_quintal": "Modal Price (₹/Q)",
                    "min": "Min Price (₹/Q)",
                    "max": "Max Price (₹/Q)",
                    "msp": "MSP (₹/Q)"
                }
            ),
            hide_index=True,
            use_container_width=True
        )

    with exp_tab2:
        cost_df = load_cost_dataset()
        cost_state_filter = st.selectbox("Filter Cost Data by State:", ["All States"] + available_states)
        if cost_state_filter != "All States":
            filtered_cost = cost_df[cost_df["state_name"] == cost_state_filter]
        else:
            filtered_cost = cost_df

        st.dataframe(
            filtered_cost[["crop_name", "state_name", "cul_cost_c2", "prod_cost_c2", "derived_yield", "opr_cost_seed", "opr_cost_fertilizer"]].rename(
                columns={
                    "crop_name": "Crop",
                    "state_name": "State",
                    "cul_cost_c2": "C2 Cultivation Cost (₹/ha)",
                    "prod_cost_c2": "Production Cost (₹/Q)",
                    "derived_yield": "Survey Yield (Q/ha)",
                    "opr_cost_seed": "Seed Cost (₹/ha)",
                    "opr_cost_fertilizer": "Fertilizer Cost (₹/ha)"
                }
            ).head(50),
            hide_index=True,
            use_container_width=True
        )

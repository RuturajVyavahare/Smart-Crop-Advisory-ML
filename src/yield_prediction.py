"""
Crop Yield Prediction Module.
Trains Random Forest and XGBoost regressors on crop_yield_dataset.csv.
Evaluates R2, RMSE, MAE, computes feature importances and actual vs predicted metrics,
and provides a production-ready YieldPredictor inference class.
"""

import os
import joblib
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.preprocessing import (
    get_yield_splits, YIELD_NUMERICAL_FEATURES, YIELD_CATEGORICAL_FEATURES,
    load_and_clean_yield_data, build_yield_preprocessor
)
from src.crop_mapping import (
    normalize_crop_name, map_rec_crop_to_yield_crop, get_crop_metadata
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "yield_prediction_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "yield_prediction_preprocessor.pkl")

# Benchmarks for typical yields (kg/ha) across all supported crops
HISTORICAL_YIELD_BENCHMARKS = {
    "Rice": {"min": 600.0, "median": 1600.0, "max": 4500.0},
    "Wheat": {"min": 800.0, "median": 1800.0, "max": 5200.0},
    "Maize": {"min": 500.0, "median": 1650.0, "max": 4800.0},
    "Chickpea": {"min": 350.0, "median": 750.0, "max": 2000.0},
    "Pigeonpea": {"min": 300.0, "median": 780.0, "max": 1800.0},
    "Cotton": {"min": 100.0, "median": 260.0, "max": 850.0},
    "Sugarcane": {"min": 25000.0, "median": 55000.0, "max": 95000.0},
    "Minor Pulses": {"min": 250.0, "median": 530.0, "max": 1400.0},
    "Groundnut": {"min": 400.0, "median": 1050.0, "max": 2800.0},
    "Soybean": {"min": 400.0, "median": 980.0, "max": 2400.0},
    "Barley": {"min": 600.0, "median": 1500.0, "max": 3800.0},
    "Pearl Millet": {"min": 300.0, "median": 850.0, "max": 2200.0},
    "Sorghum (Kharif)": {"min": 300.0, "median": 850.0, "max": 2500.0},
    "Sunflower": {"min": 350.0, "median": 920.0, "max": 2100.0},
    "Sesamum": {"min": 150.0, "median": 330.0, "max": 900.0}
}

def train_and_evaluate_yield_models(
    max_samples: int = 50000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains Random Forest and XGBoost regressors on crop_yield_dataset.csv.
    Uses representative sample for optimal convergence speed, compares R2 and RMSE,
    saves the winning model, and returns performance evaluation.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_and_clean_yield_data(max_samples=max_samples, random_state=random_state)
    X_train, X_test, y_train, y_test, preprocessor = get_yield_splits(
        df=df, test_size=0.2, random_state=random_state
    )

    # 1. Train Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=75,
        max_depth=14,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_r2 = float(r2_score(y_test, rf_preds))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    rf_mae = float(mean_absolute_error(y_test, rf_preds))

    # 2. Train XGBoost Regressor
    xgb_model = XGBRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        random_state=random_state,
        n_jobs=1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    xgb_r2 = float(r2_score(y_test, xgb_preds))
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_preds)))
    xgb_mae = float(mean_absolute_error(y_test, xgb_preds))

    # Select best model based on R2 score
    if xgb_r2 >= rf_r2:
        best_name = "XGBoost Regressor"
        best_model = xgb_model
        best_preds = xgb_preds
        best_metrics = {"r2_score": xgb_r2, "rmse": xgb_rmse, "mae": xgb_mae}
        raw_importances = xgb_model.feature_importances_
    else:
        best_name = "Random Forest Regressor"
        best_model = rf_model
        best_preds = rf_preds
        best_metrics = {"r2_score": rf_r2, "rmse": rf_rmse, "mae": rf_mae}
        raw_importances = rf_model.feature_importances_

    # Extract feature names from ColumnTransformer
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(YIELD_CATEGORICAL_FEATURES).tolist()
    all_feature_names = YIELD_NUMERICAL_FEATURES + cat_feature_names

    # Top 15 Feature Importances
    feat_imp_pairs = sorted(zip(all_feature_names, raw_importances.tolist()), key=lambda x: x[1], reverse=True)[:15]
    top_feature_importances = {name: round(score, 5) for name, score in feat_imp_pairs}

    # Sample Actual vs Predicted points for visualization
    sample_indices = np.random.choice(len(y_test), min(300, len(y_test)), replace=False)
    actual_vs_predicted = [
        {"actual": round(float(y_test[i]), 2), "predicted": round(float(best_preds[i]), 2)}
        for i in sample_indices
    ]

    # Save artifacts
    joblib.dump(best_model, MODEL_PATH)
    preprocessor_bundle = {
        "preprocessor": preprocessor,
        "numerical_features": YIELD_NUMERICAL_FEATURES,
        "categorical_features": YIELD_CATEGORICAL_FEATURES,
        "feature_names": all_feature_names,
        "model_name": best_name
    }
    joblib.dump(preprocessor_bundle, PREPROCESSOR_PATH)

    return {
        "rf_metrics": {"r2_score": rf_r2, "rmse": rf_rmse, "mae": rf_mae},
        "xgb_metrics": {"r2_score": xgb_r2, "rmse": xgb_rmse, "mae": xgb_mae},
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "top_feature_importances": top_feature_importances,
        "actual_vs_predicted": actual_vs_predicted,
        "test_sample_count": len(y_test),
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH
    }

class YieldPredictor:
    """
    Inference helper for Yield Prediction.
    Loads saved model and preprocessor to predict expected agricultural yield.
    """
    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        self.model_path = model_path or MODEL_PATH
        self.preprocessor_path = preprocessor_path or PREPROCESSOR_PATH
        self.model = None
        self.preprocessor = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Model or preprocessor file not found at {self.model_path}. "
                "Please run train_models.py first."
            )
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)["preprocessor"]

    def predict(
        self,
        crop: str,
        season: str = "Kharif",
        soil_type: str = "Vertisols",
        state: str = "Maharashtra",
        temperature: float = 26.0,
        rainfall: float = 800.0,
        humidity: float = 65.0,
        ph: float = 6.8,
        nitrogen: float = 60.0,
        phosphorus: float = 35.0,
        potassium: float = 40.0,
        fertilizer: float = 75.0,
        pesticide: float = 0.35
    ) -> Dict[str, Any]:
        """
        Predicts expected crop yield in kg/hectare.
        Returns predicted yield, unit conversions (quintals/ha, kg/acre), and comparative benchmarks.
        """
        canon_crop = normalize_crop_name(crop)
        meta = get_crop_metadata(canon_crop)
        yield_crop_name = map_rec_crop_to_yield_crop(canon_crop)

        # Check if crop has horticultural benchmark or ML dataset representation
        if yield_crop_name is None:
            # Horticultural crop: apply ICAR agronomic response scaling
            base_yield = float(meta.get("horticultural_yield_kg_ha", 12000.0))
            # Weather and soil scaling factors
            temp_factor = 1.0 - min(abs(temperature - 25.0) * 0.015, 0.25)
            rain_factor = 1.0 - min(abs(rainfall - 900.0) * 0.0002, 0.25)
            soil_factor = 1.05 if "Vertisols" in soil_type or "Inceptisols" in soil_type else 0.95
            predicted_yield_ha = round(base_yield * temp_factor * rain_factor * soil_factor, 2)
            method = "Horticultural Agronomic Calibration"
            benchmark_key = meta.get("display_name", crop.title())
        else:
            # ML Model Pipeline Inference
            df_input = pd.DataFrame([{
                "Crop": yield_crop_name,
                "Season": season,
                "Soil Type": soil_type,
                "State": state,
                "Temperature": temperature,
                "Rainfall": rainfall,
                "Humidity": humidity,
                "pH": ph,
                "Nitrogen": nitrogen,
                "Phosphorus": phosphorus,
                "Potassium": potassium,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide
            }])

            # Transform and predict
            proc_input = self.preprocessor.transform(df_input)
            raw_prediction = float(self.model.predict(proc_input)[0])
            predicted_yield_ha = max(50.0, round(raw_prediction, 2))
            method = "Trained ML Regressor"
            benchmark_key = yield_crop_name

        # Converted units
        yield_q_ha = round(predicted_yield_ha / 100.0, 2)
        yield_kg_acre = round(predicted_yield_ha / 2.47105, 2)
        yield_q_acre = round(yield_kg_acre / 100.0, 2)

        # Benchmark comparison
        benchmark = HISTORICAL_YIELD_BENCHMARKS.get(
            benchmark_key,
            {"min": predicted_yield_ha * 0.5, "median": predicted_yield_ha, "max": predicted_yield_ha * 1.8}
        )

        return {
            "crop": crop,
            "mapped_yield_crop": yield_crop_name or "Horticultural Proxy",
            "prediction_method": method,
            "predicted_yield_kg_ha": predicted_yield_ha,
            "predicted_yield_quintals_ha": yield_q_ha,
            "predicted_yield_kg_acre": yield_kg_acre,
            "predicted_yield_quintals_acre": yield_q_acre,
            "historical_min_kg_ha": benchmark["min"],
            "historical_median_kg_ha": benchmark["median"],
            "historical_max_kg_ha": benchmark["max"]
        }

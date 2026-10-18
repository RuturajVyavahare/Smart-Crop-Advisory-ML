"""
End-to-End Model Training and Evaluation Pipeline.
Trains Random Forest and XGBoost for:
1. Crop Recommendation (Multi-class Classification)
2. Crop Yield Prediction (Continuous Regression)

Saves best trained models, preprocessing artifacts, confusion matrices,
feature importances, and actual-vs-predicted plots.
"""

import os
import sys
import time
import json
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.crop_recommendation import train_and_evaluate_crop_models, CropRecommender
from src.yield_prediction import train_and_evaluate_yield_models, YieldPredictor
from src.price_lookup import get_crop_price
from src.cost_lookup import get_cultivation_cost
from src.profit_calculation import calculate_farm_financials, format_inr

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "models")

def plot_confusion_matrix(cm_data, classes, output_path):
    """Generates and saves high-resolution confusion matrix heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        np.array(cm_data),
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )
    plt.title("Crop Recommendation Model — Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Crop", fontsize=12)
    plt.ylabel("Actual Crop", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Saved Confusion Matrix to {output_path}")

def plot_crop_rec_feature_importance(importance_dict, output_path):
    """Plots feature importance for Crop Recommendation."""
    plt.figure(figsize=(9, 5))
    features = list(importance_dict.keys())
    scores = list(importance_dict.values())
    sorted_idx = np.argsort(scores)[::-1]
    
    sorted_feats = [features[i] for i in sorted_idx]
    sorted_scores = [scores[i] for i in sorted_idx]

    sns.barplot(x=sorted_scores, y=sorted_feats, palette="crest")
    plt.title("Crop Recommendation — Feature Importance Analysis", fontsize=13, pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Soil / Environmental Feature", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Saved Crop Recommendation Feature Importance to {output_path}")

def plot_yield_feature_importance(importance_dict, output_path):
    """Plots top feature importances for Yield Prediction."""
    plt.figure(figsize=(10, 6))
    features = list(importance_dict.keys())
    scores = list(importance_dict.values())
    
    sns.barplot(x=scores, y=features, palette="viridis")
    plt.title("Yield Prediction Model — Top 15 Feature Importances", fontsize=13, pad=12)
    plt.xlabel("Relative Feature Importance Score", fontsize=11)
    plt.ylabel("Predictor Variable", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Saved Yield Feature Importance to {output_path}")

def plot_actual_vs_predicted(samples, output_path):
    """Plots Actual vs Predicted yield scatter plot."""
    actuals = [s["actual"] for s in samples]
    preds = [s["predicted"] for s in samples]

    plt.figure(figsize=(8, 7))
    plt.scatter(actuals, preds, alpha=0.6, edgecolors="k", color="#2E7D32", s=40)
    
    # 45-degree reference line
    min_val = min(min(actuals), min(preds))
    max_val = max(max(actuals), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal 1:1 Line")
    
    plt.title("Yield Prediction Model — Actual vs Predicted Yield (kg/ha)", fontsize=13, pad=12)
    plt.xlabel("Actual Ground-Truth Yield (kg/ha)", fontsize=11)
    plt.ylabel("Predicted Machine Learning Yield (kg/ha)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f" Saved Actual vs Predicted plot to {output_path}")

def main():
    print("=" * 70)
    print("      SMART CROP ADVISORY SYSTEM — MODEL TRAINING PIPELINE")
    print("=" * 70)
    start_total = time.time()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. CROP RECOMMENDATION MODEL TRAINING
    # ---------------------------------------------------------
    print("\n[STEP 1/2] Training Crop Recommendation Models...")
    t0 = time.time()
    rec_results = train_and_evaluate_crop_models(random_state=42)
    rec_time = time.time() - t0

    print(f"  Training & Evaluation completed in {rec_time:.2f} seconds.")
    print("  Model Comparison:")
    print(f"    - Random Forest: Accuracy={rec_results['rf_metrics']['accuracy']*100:.2f}%, F1={rec_results['rf_metrics']['f1_score']:.4f}")
    print(f"    - XGBoost:       Accuracy={rec_results['xgb_metrics']['accuracy']*100:.2f}%, F1={rec_results['xgb_metrics']['f1_score']:.4f}")
    print(f"   SELECTED WINNER: {rec_results['best_model_name']} (Accuracy: {rec_results['best_metrics']['accuracy']*100:.2f}%)")

    # Generate Visualizations for Crop Recommendation
    cm_path = os.path.join(ARTIFACTS_DIR, "crop_rec_confusion_matrix.png")
    feat_rec_path = os.path.join(ARTIFACTS_DIR, "crop_rec_feature_importance.png")
    plot_confusion_matrix(rec_results["confusion_matrix"], rec_results["classes"], cm_path)
    plot_crop_rec_feature_importance(rec_results["feature_importances"], feat_rec_path)

    # ---------------------------------------------------------
    # 2. CROP YIELD PREDICTION MODEL TRAINING
    # ---------------------------------------------------------
    print("\n[STEP 2/2] Training Crop Yield Prediction Regressors...")
    t0 = time.time()
    yield_results = train_and_evaluate_yield_models(max_samples=50000, random_state=42)
    yield_time = time.time() - t0

    print(f"  Training & Evaluation completed in {yield_time:.2f} seconds.")
    print("  Model Comparison:")
    print(f"    - Random Forest Regressor: R²={yield_results['rf_metrics']['r2_score']:.4f}, RMSE={yield_results['rf_metrics']['rmse']:.2f}, MAE={yield_results['rf_metrics']['mae']:.2f}")
    print(f"    - XGBoost Regressor:       R²={yield_results['xgb_metrics']['r2_score']:.4f}, RMSE={yield_results['xgb_metrics']['rmse']:.2f}, MAE={yield_results['xgb_metrics']['mae']:.2f}")
    print(f"   SELECTED WINNER: {yield_results['best_model_name']} (R²: {yield_results['best_metrics']['r2_score']:.4f}, RMSE: {yield_results['best_metrics']['rmse']:.2f} kg/ha)")

    # Generate Visualizations for Yield Prediction
    yield_feat_path = os.path.join(ARTIFACTS_DIR, "yield_feature_importance.png")
    act_pred_path = os.path.join(ARTIFACTS_DIR, "yield_actual_vs_predicted.png")
    plot_yield_feature_importance(yield_results["top_feature_importances"], yield_feat_path)
    plot_actual_vs_predicted(yield_results["actual_vs_predicted"], act_pred_path)

    # Save summary metrics json for UI & verification
    summary_metrics = {
        "crop_recommendation": {
            "best_model": rec_results["best_model_name"],
            "accuracy": round(rec_results["best_metrics"]["accuracy"] * 100, 2),
            "precision": round(rec_results["best_metrics"]["precision"] * 100, 2),
            "recall": round(rec_results["best_metrics"]["recall"] * 100, 2),
            "f1_score": round(rec_results["best_metrics"]["f1_score"], 4),
            "rf_accuracy": round(rec_results["rf_metrics"]["accuracy"] * 100, 2),
            "xgb_accuracy": round(rec_results["xgb_metrics"]["accuracy"] * 100, 2),
        },
        "yield_prediction": {
            "best_model": yield_results["best_model_name"],
            "r2_score": round(yield_results["best_metrics"]["r2_score"], 4),
            "rmse": round(yield_results["best_metrics"]["rmse"], 2),
            "mae": round(yield_results["best_metrics"]["mae"], 2),
            "rf_r2": round(yield_results["rf_metrics"]["r2_score"], 4),
            "xgb_r2": round(yield_results["xgb_metrics"]["r2_score"], 4),
        }
    }
    metrics_path = os.path.join(ARTIFACTS_DIR, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"\n Saved training metrics summary to {metrics_path}")

    # ---------------------------------------------------------
    # 3. END-TO-END VERIFICATION INFERENCE
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print(" VERIFYING END-TO-END INFERENCE INTEGRATION...")
    print("-" * 70)
    
    # 1. Recommendation
    recommender = CropRecommender()
    rec_out = recommender.predict(
        n=90.0, p=42.0, k=43.0,
        temperature=21.0, humidity=82.0, ph=6.5, rainfall=202.0
    )
    rec_crop = rec_out["primary_crop"]
    print(f"  1. Recommended Crop: {rec_out['display_name']} (Confidence: {rec_out['confidence_percent']}%)")

    # 2. Yield Prediction
    predictor = YieldPredictor()
    yield_out = predictor.predict(
        crop=rec_crop, season="Kharif", soil_type="Vertisols", state="Maharashtra",
        temperature=21.0, rainfall=202.0, humidity=82.0, ph=6.5,
        nitrogen=90.0, phosphorus=42.0, potassium=43.0, fertilizer=100.0, pesticide=0.4
    )
    pred_yield_ha = yield_out["predicted_yield_kg_ha"]
    print(f"  2. Predicted Yield: {pred_yield_ha:.2f} kg/ha ({yield_out['predicted_yield_quintals_acre']:.2f} quintals/acre)")

    # 3. Price Lookup
    price_info = get_crop_price(rec_crop)
    modal_price = price_info["modal_rs_per_quintal"]
    print(f"  3. Mandi Market Price: {format_inr(modal_price)} / Quintal ({price_info['matched_crop_name']})")

    # 4. Cost Lookup
    cost_info = get_cultivation_cost(rec_crop, state_name="Maharashtra")
    c2_cost_ha = cost_info["cul_cost_c2_per_ha"]
    print(f"  4. Cultivation Cost (C2): {format_inr(c2_cost_ha)} / Hectare ({format_inr(cost_info['cul_cost_c2_per_acre'])} / Acre)")

    # 5. Profit & Revenue
    farm_area = 2.5
    area_unit = "acre"
    financials = calculate_farm_financials(
        yield_kg_per_ha=pred_yield_ha,
        farm_area=farm_area,
        area_unit=area_unit,
        price_per_quintal=modal_price,
        cost_per_ha=c2_cost_ha
    )
    print(f"  5. Farm Area: {farm_area} {area_unit.title()}s ({financials['area_hectares']} ha)")
    print(f"     - Harvest Production: {financials['total_production_quintals']} Quintals ({financials['total_production_kg']} kg)")
    print(f"     - Estimated Revenue:  {financials['formatted_revenue']}")
    print(f"     - Cultivation Cost:   {financials['formatted_cost']}")
    print(f"     - Estimated Profit:   {financials['formatted_profit']} (ROI: {financials['roi_percent']}%)")
    print(f"     - Profit per Acre:    {financials['formatted_profit_per_acre']}")

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f" ALL MODELS TRAINED, EVALUATED & VERIFIED SUCCESSFULLY in {total_time:.2f}s!")
    print("=" * 70)

if __name__ == "__main__":
    main()

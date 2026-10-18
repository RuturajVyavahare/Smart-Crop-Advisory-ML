"""
Crop Recommendation Model Training and Inference Module.
Trains Random Forest and XGBoost classifiers, evaluates cross-metrics,
selects the superior model, computes feature importances and confusion matrices,
and provides an inference class for real-time recommendations.
"""

import os
import joblib
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from src.preprocessing import (
    get_crop_rec_splits, CROP_REC_FEATURES, load_and_clean_crop_rec_data
)
from src.crop_mapping import get_crop_metadata, normalize_crop_name

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_recommendation_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "crop_recommendation_preprocessor.pkl")

def train_and_evaluate_crop_models(
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains both Random Forest and XGBoost classifiers on Crop_recommendation.csv.
    Evaluates both models, selects the best, saves artifacts, and returns comprehensive metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_and_clean_crop_rec_data()
    X_train, X_test, y_train, y_test, label_encoder, scaler = get_crop_rec_splits(
        df=df, test_size=0.2, random_state=random_state
    )

    classes = label_encoder.classes_

    # 1. Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=3,
        random_state=random_state,
        n_jobs=1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds, average="weighted", zero_division=0)
    rf_rec = recall_score(y_test, rf_preds, average="weighted", zero_division=0)
    rf_f1 = f1_score(y_test, rf_preds, average="weighted", zero_division=0)

    # 2. Train XGBoost Classifier
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_prec = precision_score(y_test, xgb_preds, average="weighted", zero_division=0)
    xgb_rec = recall_score(y_test, xgb_preds, average="weighted", zero_division=0)
    xgb_f1 = f1_score(y_test, xgb_preds, average="weighted", zero_division=0)

    # Compare and select best
    if xgb_f1 > rf_f1:
        best_name = "XGBoost Classifier"
        best_model = xgb_model
        best_preds = xgb_preds
        best_metrics = {"accuracy": xgb_acc, "precision": xgb_prec, "recall": xgb_rec, "f1_score": xgb_f1}
        best_importances = dict(zip(CROP_REC_FEATURES, xgb_model.feature_importances_.tolist()))
    else:
        best_name = "Random Forest Classifier"
        best_model = rf_model
        best_preds = rf_preds
        best_metrics = {"accuracy": rf_acc, "precision": rf_prec, "recall": rf_rec, "f1_score": rf_f1}
        best_importances = dict(zip(CROP_REC_FEATURES, rf_model.feature_importances_.tolist()))

    cm = confusion_matrix(y_test, best_preds)
    cr = classification_report(y_test, best_preds, target_names=classes, output_dict=True, zero_division=0)

    # Save artifacts
    joblib.dump(best_model, MODEL_PATH)
    preprocessor_bundle = {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": CROP_REC_FEATURES,
        "classes": classes.tolist(),
        "model_name": best_name
    }
    joblib.dump(preprocessor_bundle, PREPROCESSOR_PATH)

    return {
        "rf_metrics": {"accuracy": rf_acc, "precision": rf_prec, "recall": rf_rec, "f1_score": rf_f1},
        "xgb_metrics": {"accuracy": xgb_acc, "precision": xgb_prec, "recall": xgb_rec, "f1_score": xgb_f1},
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "feature_importances": best_importances,
        "confusion_matrix": cm.tolist(),
        "classes": classes.tolist(),
        "classification_report": cr,
        "test_sample_count": len(y_test),
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH
    }

class CropRecommender:
    """
    Inference helper for Crop Recommendation.
    Loads saved model and preprocessor to generate top recommendations with probabilities.
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
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict(
        self,
        n: float,
        p: float,
        k: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Recommends best crops given soil nutrients and environmental parameters.
        Returns top recommendation, top-k ranking with probabilities, and agronomic insights.
        """
        scaler = self.preprocessor["scaler"]
        encoder = self.preprocessor["label_encoder"]
        classes = self.preprocessor["classes"]

        input_arr = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_arr)

        probabilities = self.model.predict_proba(input_scaled)[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        recommendations: List[Dict[str, Any]] = []
        for idx in top_indices:
            crop_label = str(classes[idx])
            prob = float(probabilities[idx])
            meta = get_crop_metadata(crop_label)
            recommendations.append({
                "crop": crop_label,
                "display_name": meta.get("display_name", crop_label.title()),
                "category": meta.get("category", "Field Crop"),
                "confidence_score": round(prob * 100.0, 2),
                "probability": round(prob, 4),
                "description": meta.get("description", "")
            })

        primary = recommendations[0]

        # Soil nutrient diagnostic
        nutrient_notes = []
        if n < 40:
            nutrient_notes.append("Low Soil Nitrogen (N): Consider legume cover crops or urea supplementation.")
        elif n > 120:
            nutrient_notes.append("High Soil Nitrogen (N): Suitable for heavy-foliage crops; minimize nitrogenous fertilizers.")
            
        if p < 25:
            nutrient_notes.append("Low Phosphorus (P): Superphosphate (SSP/DAP) recommended for strong root development.")
        if k < 25:
            nutrient_notes.append("Low Potassium (K): Muriate of Potash (MOP) recommended for disease resistance.")

        if ph < 5.5:
            nutrient_notes.append("Acidic Soil (pH < 5.5): Agricultural lime application advised.")
        elif ph > 8.0:
            nutrient_notes.append("Alkaline Soil (pH > 8.0): Gypsum or organic compost advised to balance pH.")

        return {
            "primary_crop": primary["crop"],
            "display_name": primary["display_name"],
            "confidence_percent": primary["confidence_score"],
            "category": primary["category"],
            "description": primary["description"],
            "recommendations": recommendations,
            "soil_advisory_notes": nutrient_notes,
            "model_used": self.preprocessor.get("model_name", "Classification Model")
        }

# SMART CROP ADVISORY SYSTEM
## Machine Learning Based Crop Recommendation, Yield Prediction & Profit Estimation
### Complete Academic Project Report & Technical Specification Document

---

## 📋 TABLE OF CONTENTS
1. [Abstract](#1-abstract)
2. [Introduction & Problem Statement](#2-introduction--problem-statement)
3. [Literature Survey & Background](#3-literature-survey--background)
4. [System Architecture & Design](#4-system-architecture--design)
5. [Datasets Description & Preprocessing](#5-datasets-description--preprocessing)
6. [Machine Learning Methodology](#6-machine-learning-methodology)
7. [Economic & Profit Estimation Engine](#7-economic--profit-estimation-engine)
8. [Experimental Results & Evaluation](#8-experimental-results--evaluation)
9. [User Interface & Software Implementation](#9-user-interface--software-implementation)
10. [Testing & Quality Assurance](#10-testing--quality-assurance)
11. [Conclusion & Future Scope](#11-conclusion--future-scope)
12. [References](#12-references)

---

## 1. ABSTRACT

Agriculture is the backbone of the Indian economy, employing over 50% of the workforce yet often challenged by sub-optimal crop selection, volatile market prices, and unpredictable yields. Traditional farming decisions depend on intuition or historical convention rather than empirical soil health and climatic indicators.

This project presents the **Smart Crop Advisory System**, a unified decision-support engine combining machine learning with real-world agricultural economics. The system solves three interrelated challenges:
1. **Which crop to sow?** A multi-class classifier analyzes 7 soil and climatic parameters (N, P, K, temperature, humidity, pH, rainfall) across 22 crops, with **Random Forest achieving 99.55% test accuracy**.
2. **What will be the expected harvest?** A continuous regression engine predicts yield in kg/hectare across 21 crops and 20 states using 198,000+ historical records, with **XGBoost Regressor achieving an $R^2$ of 0.8801**.
3. **What is the commercial viability?** An automated economic engine cross-references live APMC Mandi modal prices (from a 122-crop index) and Directorate of Economics & Statistics (DES/CACP) state-wise Comprehensive C2 Cultivation Costs.

The integrated pipeline calculates total production, gross revenue, cultivation expenses, net profit, return on investment (ROI %), and break-even thresholds. The entire pipeline is deployed through an interactive Streamlit application supporting agro-climatic presets, scenario simulations, and automated report generation.

---

## 2. INTRODUCTION & PROBLEM STATEMENT

### 2.1 Background
Indian agriculture is characterized by high agro-ecological diversity, spanning 15 distinct agro-climatic zones. Farmers face complex multi-factorial decisions involving soil fertility, monsoon patterns, market accessibility, and input costs. 

### 2.2 Problem Statement
1. **Soil-Crop Mismatch**: Farmers frequently plant traditional crops unsuited to current soil nutrient deficits (Nitrogen, Phosphorus, Potassium) or shifted monsoon rainfall patterns, resulting in crop failure or wasted fertilizers.
2. **Yield Uncertainty**: Lack of data-driven yield estimation prevents farmers from planning storage, labor, and forward-selling contracts.
3. **Price & Cost Disconnect**: Existing crop recommendation systems output only a biological crop name without commercial context. A high-yield crop may yield a net loss if local mandi prices crash or if cultivation costs exceed revenues.
4. **Disjointed Tools**: Previous software either only predicts crops or only tracks prices; none integrates biological suitability with commercial viability into a unified advisory workflow.

### 2.3 Objectives
- Develop and evaluate multi-class classification models (Random Forest vs. XGBoost) for crop recommendation.
- Develop and evaluate non-linear regression models (Random Forest vs. XGBoost) for crop yield estimation.
- Integrate real-world APMC Mandi market prices and CACP Comprehensive C2 cultivation costs.
- Build an economic calculation engine providing production volumes, revenues, net profits, and break-even points in Indian Rupees (INR).
- Deliver a responsive, professional Streamlit web application with agro-climatic presets, scenario simulations, and CSV export capabilities.

---

## 3. LITERATURE SURVEY & BACKGROUND

| Author / Study | Methodology | Features Used | Limitations |
| :--- | :--- | :--- | :--- |
| **Kumar et al. (2020)** | Decision Tree & Naive Bayes | Soil N, P, K, pH | Low accuracy (<88%); no yield prediction or economic analysis. |
| **Rajak et al. (2021)** | Random Forest Classifier | Climate variables | Only suggested crop; omitted market price and cultivation cost. |
| **Patil & Shrikant (2022)**| Multiple Linear Regression | Rainfall, Area | Linear assumption failed on non-linear agricultural interactions ($R^2 < 0.65$). |
| **Proposed System** | **Random Forest (99.55%) + XGBoost Regressor ($R^2=0.88$)** | **7 Soil/Climate + 13 Agronomic variables + 122 Mandi Prices + CACP C2 Costs** | **End-to-end biological and commercial advisory with scenario simulations.** |

---

## 4. SYSTEM ARCHITECTURE & DESIGN

### 4.1 End-to-End Workflow Diagram

```
[ Farmer Inputs ]
  ├── Soil Nutrients: Nitrogen (N), Phosphorus (P), Potassium (K), pH
  ├── Environment: Temperature (°C), Humidity (%), Rainfall (mm)
  ├── Geography: State, Soil Type, Cropping Season
  └── Landholding: Farm Area (Acres / Hectares)
           │
           ▼
[ Step 1: Crop Recommendation Classifier ]
  ├── Random Forest Model (Trained on 2,200 records, 22 crops)
  └── Output: Primary Crop Match + Top-3 Alternatives + Confidence Scores
           │
           ▼
[ Step 2: Crop Name Normalization & Canonical Mapping ]
  └── Maps canonical key across Yield, Price, and Cost datasets
           │
           ▼
[ Step 3: Yield Prediction Regressor ]
  ├── XGBoost Regressor (Trained on 198,369 records, 21 crops, 20 states)
  └── Output: Expected Yield (kg/ha, Quintals/ha, Quintals/Acre)
           │
           ▼
[ Step 4: Mandi Market Price Lookup ]
  ├── Database: india-crop-prices.csv (122 crops)
  └── Output: Modal Price (₹/Quintal), Min/Max Range, MSP
           │
           ▼
[ Step 5: CACP Cultivation Cost Retrieval ]
  ├── Database: cost-of-cultivation.csv (3,880 state records)
  └── Output: C2 Comprehensive Cost (₹/ha, ₹/Acre), Seed, Fertilizer, Labor breakdown
           │
           ▼
[ Step 6: Agricultural Economics Engine ]
  ├── Total Production = Yield × Area
  ├── Total Revenue = Production × Price
  ├── Total Cost = Area × C2 Cost
  ├── Net Profit = Revenue - Cost
  ├── Return on Investment (ROI %) = (Profit / Cost) × 100
  └── Break-Even Yield & Break-Even Market Price
           │
           ▼
[ Step 7: Streamlit Interactive Advisory Dashboard ]
```

---

## 5. DATASETS DESCRIPTION & PREPROCESSING

The project utilizes **four verified agricultural datasets**:

### 5.1 Dataset 1: Crop Recommendation (`Crop_recommendation.csv`)
- **Observations**: 2,200 samples (100 samples per crop class).
- **Target**: 22 crops (`rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`).
- **Features**:
  - `N`: Nitrogen ratio in soil (0 to 140 kg/ha)
  - `P`: Phosphorus ratio in soil (5 to 145 kg/ha)
  - `K`: Potassium ratio in soil (5 to 205 kg/ha)
  - `temperature`: Temperature in °C (8.8°C to 43.7°C)
  - `humidity`: Relative humidity in % (14.3% to 99.9%)
  - `ph`: Soil pH value (3.5 to 9.9)
  - `rainfall`: Rainfall in mm (20.2 mm to 298.6 mm)

### 5.2 Dataset 2: Crop Yield Historical Data (`crop_yield_dataset.csv`)
- **Observations**: 198,369 historical records spanning 1966 to 2017 across 309 districts and 20 Indian states.
- **Categorical Features**: `Crop` (21 major field crops), `Season` (Kharif, Rabi, Whole Year), `Soil Type` (Alfisols, Aridisols, Inceptisols, Mollisols, Vertisols, Entisols), `State` (extracted from Region).
- **Numerical Features**: `Temperature`, `Rainfall`, `Humidity`, `pH`, `Nitrogen`, `Phosphorus`, `Potassium`, `Fertilizer` (kg/ha), `Pesticide` (kg/ha).
- **Target**: `Yield` in kg/hectare (Range: 0.01 to 58,363 kg/ha, Median: 800 kg/ha).

### 5.3 Dataset 3: India Mandi Crop Prices (`india-crop-prices.csv`)
- **Records**: 122 commercial and horticultural agricultural commodities.
- **Attributes**: `crop`, `modal_rs_per_quintal`, `min`, `max`, `msp` (Minimum Support Price where applicable).

### 5.4 Dataset 4: Cost of Cultivation (`cost-of-cultivation.csv`)
- **Records**: 3,880 state-level agricultural survey records from the Commission for Agricultural Costs and Prices (CACP).
- **Key Metrics**:
  - `cul_cost_c2`: Comprehensive C2 Cost of Cultivation in ₹/Hectare (includes actual expenses + rent of owned land + interest on owned fixed capital + imputed family labor).
  - `prod_cost_c2`: Cost of Production in ₹/Quintal.
  - Input cost breakdowns: Seed (`opr_cost_seed`), Fertilizer (`opr_cost_fertilizer`), Labor (`opr_cost_hmn_lab`), Machinery (`opr_cost_mch_lab`).

### 5.5 Preprocessing Methodology
1. **Missing Values & Deduplication**: Complete check performed; zero nulls in feature vectors; duplicate entries removed.
2. **Anomaly Cleaning**: Negative sensor temperatures clipped to valid agronomic lower bounds (5.0°C); pH values bounded to [3.0, 10.0]; non-positive yield records purged.
3. **Region-to-State Parsing**: Standardized district-state strings (e.g., `'Anantapur, AndhraPradesh'` ➔ `'Andhra Pradesh'`).
4. **Encoding & Transformations**:
   - Classification target encoded via `LabelEncoder`.
   - Continuous inputs scaled using `StandardScaler` ($\mu=0, \sigma=1$).
   - Categorical inputs (`Crop`, `Season`, `Soil Type`, `State`) encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` within a scikit-learn `ColumnTransformer`.

---

## 6. MACHINE LEARNING METHODOLOGY

### 6.1 Algorithm Selection: Random Forest vs. XGBoost
Both tree-based ensemble algorithms were evaluated across both tasks:

#### Random Forest (Bagging Ensemble)
Constructs an ensemble of de-correlated decision trees:
$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$
Reduces variance and eliminates overfitting through bootstrap aggregation and random feature sub-spacing.

#### XGBoost (Gradient Boosting)
Minimizes a regularized objective function sequentially:
$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$
Where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ penalizes tree complexity and leaf weights.

---

## 7. ECONOMIC & PROFIT ESTIMATION ENGINE

### 7.1 Mathematical Formulations

1. **Unit Conversion**:
   $$\text{Area (Hectares)} = \frac{\text{Area (Acres)}}{2.47105}$$
   $$\text{Area (Acres)} = \text{Area (Hectares)} \times 2.47105$$

2. **Harvest Production**:
   $$\text{Yield (Quintals/ha)} = \frac{\text{Yield (kg/ha)}}{100}$$
   $$\text{Total Production (Quintals)} = \text{Yield (Quintals/ha)} \times \text{Area (ha)}$$

3. **Gross Revenue**:
   $$\text{Gross Revenue (₹)} = \text{Total Production (Quintals)} \times \text{Modal Mandi Price (₹/Quintal)}$$

4. **Cultivation Cost**:
   $$\text{Total Cost (₹)} = \text{Area (ha)} \times \text{C2 Cost of Cultivation (₹/ha)}$$

5. **Net Profit & ROI**:
   $$\text{Net Profit (₹)} = \text{Gross Revenue} - \text{Total Cost}$$
   $$\text{Profit Per Acre (₹)} = \frac{\text{Net Profit}}{\text{Area (Acres)}}$$
   $$\text{Return on Investment (ROI \%)} = \left(\frac{\text{Net Profit}}{\text{Total Cost}}\right) \times 100$$

6. **Break-Even Analysis**:
   $$\text{Break-Even Yield (kg/ha)} = \frac{\text{Cost per Hectare (₹)}}{\text{Price per kg (₹)}}$$
   $$\text{Break-Even Market Price (₹/Quintal)} = \frac{\text{Total Cost (₹)}}{\text{Total Production (Quintals)}}$$

---

## 8. EXPERIMENTAL RESULTS & EVALUATION

### 8.1 Crop Recommendation Model Performance

| Metric | Random Forest Classifier | XGBoost Classifier | Selected Model |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **99.55%** | 99.09% | **Random Forest** 🏆 |
| **Weighted Precision** | **99.57%** | 99.15% | **Random Forest** |
| **Weighted Recall** | **99.55%** | 99.09% | **Random Forest** |
| **Weighted F1-Score**| **0.9955** | 0.9909 | **Random Forest** |

- **Confusion Matrix**: High diagonal density across all 22 classes; minimal misclassifications between closely related pulses.
- **Feature Importance**: Humidity (22.8%), Rainfall (21.4%), Phosphorus (16.2%), Potassium (15.8%), Nitrogen (11.7%), Temperature (7.1%), pH (5.0%).

### 8.2 Crop Yield Prediction Model Performance

| Metric | Random Forest Regressor | XGBoost Regressor | Selected Model |
| :--- | :---: | :---: | :---: |
| **Coefficient of Determination ($R^2$)** | 0.8757 | **0.8801** | **XGBoost Regressor** 🏆 |
| **Root Mean Squared Error (RMSE)** | 525.41 kg/ha | **516.15 kg/ha** | **XGBoost Regressor** |
| **Mean Absolute Error (MAE)** | 318.92 kg/ha | **308.29 kg/ha** | **XGBoost Regressor** |

- **Actual vs Predicted**: Scatter plot clusters tightly along the ideal $45^\circ$ line ($y=x$), demonstrating strong generalization across rainfed, irrigated, and high-yielding cash crops.

---

## 9. USER INTERFACE & SOFTWARE IMPLEMENTATION

### 9.1 Technology Stack
- **Core Language**: Python 3.11
- **ML Frameworks**: scikit-learn, XGBoost, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Web Application**: Streamlit
- **Packaging & Environment**: `uv` package manager

### 9.2 Key UI Capabilities
1. **Interactive Presets**: Gangetic Plains, Deccan Plateau, Semi-Arid Rajasthan, and Coastal Humid.
2. **Six Top KPI Cards**: Recommended Crop, Predicted Yield, Market Price, Cultivation Cost, Total Revenue, Net Profit.
3. **Interactive Simulation**: Real-time sliders adjusting market price (±40%) and yield (±40%) to simulate drought or bumper crop economic scenarios.
4. **One-Click Export**: Downloads full advisory report as a structured CSV file.

---

## 10. TESTING & QUALITY ASSURANCE

Automated unit and integration testing implemented via `pytest`:
- **Test File**: `tests/test_pipeline.py`
- **Total Tests**: 15 test cases
- **Pass Rate**: 100% (15/15 passed)
- **Coverage**:
  - `test_crop_normalization`: Canonical mapping across casing, spaces, and aliases.
  - `test_all_supported_crops_count`: Verifies all 22 crops metadata consistency.
  - `test_price_lookup`: Mandi price retrieval, min/max bounds, and fallback handling.
  - `test_cost_lookup`: C2 comprehensive cost retrieval and state filtering.
  - `test_unit_conversions`: Acre-hectare and kg-quintal precision.
  - `test_farm_financials_math`: Verification of revenue, cost, profit, and ROI formulas.
  - `test_crop_recommender_prediction`: Classifier inference and confidence scores.
  - `test_yield_predictor_prediction`: Regressor inference and unit conversions.
  - `test_complete_end_to_end_flow`: Full pipeline integration test from inputs to net profit.

---

## 11. CONCLUSION & FUTURE SCOPE

### 11.1 Conclusion
The **Smart Crop Advisory System** successfully bridges the gap between agricultural data science and farm economics. By achieving **99.55% accuracy** in crop recommendation and an **$R^2$ of 0.8801** in yield prediction, combined with actual Mandi prices and CACP cultivation costs, the system provides farmers with actionable, profit-maximizing advisories.

### 11.2 Future Scope
1. **IoT Sensor Integration**: Real-time soil moisture and automated NPK sensor feeds.
2. **Satellite & Remote Sensing**: Incorporating Sentinel-2 / Landsat NDVI spectral imagery.
3. **Deep Learning Weather Forecasting**: Integrating LSTM / Transformer models for 15-day localized rainfall predictions.
4. **Multilingual Voice Bot**: Speech-to-text advisory for regional vernacular communication.

---

## 12. REFERENCES
1. Directorate of Economics & Statistics (DES), Ministry of Agriculture & Farmers Welfare, Government of India. *Cost of Cultivation of Principal Crops in India*.
2. Agmarknet Portal, Agricultural Marketing Information Network. *Daily Mandi Prices and Arrivals*.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
5. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.

"""
Air Quality AQI Prediction — XGBoost Model Training
Run: python train_model.py
"""

import numpy as np
import pandas as pd
import os, joblib, json
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── 1. Load dataset ─────────────────────────────────────────────────────────
df = pd.read_csv("air_quality_dataset.csv")
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# ── 2. Feature engineering ──────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"])
df["month"]      = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["season"]     = df["month"].map(
    {12: 0, 1: 0, 2: 0,   # Winter
     3: 1, 4: 1, 5: 1,    # Spring
     6: 2, 7: 2, 8: 2,    # Monsoon
     9: 3, 10: 3, 11: 3}  # Autumn
)

# Pollution ratios and transforms (interaction features)
df["pm_ratio"]       = df["pm25"] / (df["pm10"] + 1e-6)
df["no2_so2_ratio"]  = df["no2"]  / (df["so2"]  + 1e-6)
df["wind_temp"]      = df["wind_speed"] * df["temperature"]
df["rh_temp"]        = df["humidity"]   * df["temperature"]

# Log-transform skewed pollutants (helps tree models)
for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
    df[f"log_{col}"] = np.log1p(df[col])

# AQI sub-index approximations (mirrors the formula used to build the target)
def _aqi_sub(c, bp):
    for lo_c, hi_c, lo_i, hi_i in bp:
        if lo_c <= c <= hi_c:
            return ((hi_i - lo_i) / (hi_c - lo_c)) * (c - lo_c) + lo_i
    return 500.0

pm25_bp = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
           (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
pm10_bp = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
           (255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]

df["aqi_pm25_sub"] = df["pm25"].apply(lambda x: _aqi_sub(x, pm25_bp))
df["aqi_pm10_sub"] = df["pm10"].apply(lambda x: _aqi_sub(x, pm10_bp))
df["aqi_max_sub"]  = df[["aqi_pm25_sub", "aqi_pm10_sub"]].max(axis=1)

# Squared wind (diminishing dispersion effect)
df["wind_sq"]  = df["wind_speed"] ** 2
# Total oxidant load
df["oxidants"] = df["no2"] + df["o3"]

# City label encoding
le = LabelEncoder()
df["city_enc"] = le.fit_transform(df["city"])

FEATURES = [
    "city_enc", "month", "day_of_year", "season",
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed",
    "pm_ratio", "no2_so2_ratio", "wind_temp", "rh_temp",
    "log_pm25", "log_pm10", "log_no2", "log_so2", "log_co", "log_o3",
    "aqi_pm25_sub", "aqi_pm10_sub", "aqi_max_sub",
    "wind_sq", "oxidants",
]
TARGET = "aqi"

X = df[FEATURES].values
y = df[TARGET].values

# ── 3. Train/test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# ── 4. Model ─────────────────────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=1200,
    max_depth=7,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1.5,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="mae",
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)

# ── 5. Evaluation ────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("  MODEL EVALUATION")
print("="*50)
print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")
print(f"  R2   : {r2:.4f}")
print("="*50)

# Cross-validation R²
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(
    XGBRegressor(
        n_estimators=600, max_depth=7, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, random_state=42,
        tree_method="hist", n_jobs=-1,
    ),
    X, y, cv=kf, scoring="r2", n_jobs=-1
)
print(f"  CV R2: {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print("="*50)

# ── 6. Feature importance ────────────────────────────────────────────────────
importance = dict(zip(FEATURES, model.feature_importances_.tolist()))
importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
print("\nTop feature importances:")
for feat, imp in list(importance_sorted.items())[:10]:
    print(f"  {feat:20s}: {imp:.4f}")

# ── 7. Save artifacts ────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/xgb_aqi_model.pkl")
joblib.dump(le, "model/city_encoder.pkl")

meta = {
    "features": FEATURES,
    "target": TARGET,
    "metrics": {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 4)},
    "cv_r2_mean": round(float(cv_r2.mean()), 4),
    "cv_r2_std":  round(float(cv_r2.std()), 4),
    "cities": le.classes_.tolist(),
    "feature_importance": importance_sorted,
}
with open("model/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nDONE: Model, encoder, and metadata saved to ./model/")

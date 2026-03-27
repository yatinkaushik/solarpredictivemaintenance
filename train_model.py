"""
train_model.py  — Ensemble Model with Calibrated Probabilities
==============================================================
Key improvements over v1:
  - 19 features (was 5) — adds Daily_kWh, Z-score, lag, trend, seasonality
  - VotingClassifier: RandomForest + GradientBoosting + ExtraTrees (soft vote)
  - CalibratedClassifierCV: converts raw scores to reliable probabilities
  - 70/15/15 time-based train/val/test split (no future leakage)
  - TimeSeriesSplit cross-validation on training set
  - Full evaluation report: AUC-ROC, precision, recall, F1
  - Feature importance saved alongside model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

print("🚀 Loading dataset...")

df = pd.read_csv("solar_predictive_maintenance_dataset.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

# =====================================================
# FEATURE SET — 19 features (was 5)
# =====================================================
FEATURES = [
    # ── Core environmental ──────────────────────────
    "ALLSKY_SFC_SW_DWN",       # Solar irradiance
    "T2M",                     # Temperature
    "TEMP_7DAY_AVG",           # Temp rolling avg
    "SOLAR_7DAY_AVG",          # Irradiance rolling avg
    "SOLAR_RATIO",             # Today vs 7-day baseline irradiance

    # ── Performance ratio features ──────────────────
    "Performance_Ratio",       # Core PR signal
    "PR_3DAY_AVG",             # Short-term baseline
    "PR_14DAY_AVG",            # Medium-term baseline
    "PR_30DAY_AVG",            # Long-term baseline
    "PR_7DAY_STD",             # Volatility (instability signal)
    "PR_Z_SCORE",              # Standardised deviation ← strong signal
    "PR_7DAY_TREND",           # Direction: improving or degrading?

    # ── Lag features (yesterday's state) ────────────
    "PR_LAG1",                 # Yesterday PR
    "PR_LAG3",                 # 3 days ago PR

    # ── Energy features (previously unused!) ────────
    "Daily_kWh",               # Actual solar generation  (corr = −0.47)
    "Daily_kWh_7DAY_AVG",      # Rolling generation baseline
    "kWh_elec_per_day",        # Grid electricity use     (corr = +0.41)

    # ── Seasonality ─────────────────────────────────
    "DOY_SIN",                 # Cyclical day-of-year (sin)
    "DOY_COS",                 # Cyclical day-of-year (cos)
]

# Verify all features exist
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(
        f"❌ Missing features: {missing}\n"
        "Re-run prepare_merged_dataset.py first to generate the enhanced dataset."
    )

X = df[FEATURES].ffill().bfill()
y = df["Anomaly"]

# =====================================================
# TIME-BASED SPLIT  (no data leakage)
# 70% train | 15% validation | 15% test
# =====================================================
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X.iloc[:train_end],  y.iloc[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:],    y.iloc[val_end:]

print(f"\nTrain: {len(X_train):,} rows  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
print(f"Train anomaly rate: {y_train.mean():.3f}")
print(f"Test  anomaly rate: {y_test.mean():.3f}")

# =====================================================
# BUILD ENSEMBLE
# Three diverse estimators with soft (probability) voting
# =====================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42,
)

et = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("et", et)],
    voting="soft",
)

# =====================================================
# CROSS-VALIDATION (time-series aware, on training set)
# =====================================================
print("\n📊 Running TimeSeriesSplit cross-validation (5 folds)...")
tscv = TimeSeriesSplit(n_splits=5)
cv_auc = cross_val_score(ensemble, X_train, y_train, cv=tscv, scoring="roc_auc", n_jobs=-1)
cv_ap  = cross_val_score(ensemble, X_train, y_train, cv=tscv, scoring="average_precision", n_jobs=-1)

print(f"  CV AUC-ROC:           {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  CV Avg Precision:     {cv_ap.mean():.4f} ± {cv_ap.std():.4f}")

# =====================================================
# TRAIN ENSEMBLE ON FULL TRAINING SET
# =====================================================
print("\n🏋️  Training ensemble on full training set...")
ensemble.fit(X_train, y_train)

# =====================================================
# CALIBRATE PROBABILITIES ON VALIDATION SET
# Isotonic regression maps raw ensemble scores →
# reliable, well-calibrated probabilities.
# =====================================================
print("🔧 Calibrating probabilities on validation set...")
raw_val_probs = ensemble.predict_proba(X_val)[:, 1]
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(raw_val_probs, y_val)

# =====================================================
# EVALUATE ON HELD-OUT TEST SET
# =====================================================
print("\n" + "="*55)
print("📈 TEST SET EVALUATION (final holdout — never seen during training)")
print("="*55)

y_raw_prob  = ensemble.predict_proba(X_test)[:, 1]
y_prob      = calibrator.predict(y_raw_prob)          # calibrated
y_pred      = (y_prob >= 0.5).astype(int)

auc  = roc_auc_score(y_test, y_prob)
ap   = average_precision_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print(f"\nAUC-ROC:              {auc:.4f}")
print(f"Avg Precision (AUPRC):{ap:.4f}")
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

# =====================================================
# FEATURE IMPORTANCE (from Random Forest component)
# =====================================================
rf_fitted = ensemble.estimators_[0]
importances = pd.Series(rf_fitted.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print("\n🔍 Top 10 Feature Importances (Random Forest):")
for feat, imp in importances.head(10).items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# =====================================================
# SAVE MODEL + METADATA
# =====================================================
model_payload = {
    "model": ensemble,
    "calibrator": calibrator,
    "features": FEATURES,
    "auc_roc": auc,
    "avg_precision": ap,
    "feature_importances": importances.to_dict(),
}

joblib.dump(model_payload, "model.pkl")
print(f"\n✅ Model saved as model.pkl")
print(f"   AUC-ROC: {auc:.4f}  |  AUPRC: {ap:.4f}")
print("\n🎯 READY — run: streamlit run app.py")
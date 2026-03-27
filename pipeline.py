"""
pipeline.py — Inference pipeline for Solar AI
Robust model loading with fallback, graceful feature mismatch handling,
and structured logging replacing all print() calls.
"""
import pandas as pd
import numpy as np
import joblib
import os

from config import MODEL_PATH
from logger import get_logger

log = get_logger("solar_ai.pipeline")

# ── Default features (fallback if model.pkl has no metadata) ─────────────────
_DEFAULT_FEATURES = [
    "ALLSKY_SFC_SW_DWN", "T2M", "TEMP_7DAY_AVG",
    "SOLAR_7DAY_AVG", "Performance_Ratio",
]

# ── Model loading with full fallback ─────────────────────────────────────────
def _load_model():
    """
    Loads model.pkl with three layers of safety:
    1. File existence check before attempting load
    2. Corrupt/incompatible model caught by exception
    3. Missing metadata keys handled gracefully with defaults
    Returns (model, calibrator, features, auc, ap) tuple.
    """
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model file not found at: {MODEL_PATH}")
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PATH}. "
            "Run train_model.py to generate it."
        )

    try:
        payload = joblib.load(MODEL_PATH)
        log.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        log.error(f"Failed to load model.pkl — file may be corrupt: {e}")
        raise RuntimeError(
            f"Could not load model.pkl: {e}. "
            "Delete it and re-run train_model.py."
        ) from e

    if isinstance(payload, dict):
        model      = payload.get("model")
        calibrator = payload.get("calibrator", None)
        features   = payload.get("features", _DEFAULT_FEATURES)
        auc        = payload.get("auc_roc", "N/A")
        ap         = payload.get("avg_precision", "N/A")

        if model is None:
            raise ValueError("model.pkl dict is missing 'model' key — retrain.")

        if isinstance(auc, float):
            log.info(f"AUC-ROC: {auc:.4f}  |  AUPRC: {ap:.4f}")
        log.info(f"Features: {len(features)}")

    else:
        # Legacy single-object format
        model      = payload
        calibrator = None
        features   = _DEFAULT_FEATURES
        log.warning(
            "Old model format detected (not a dict). "
            "Re-run train_model.py to upgrade to the current format."
        )
        auc, ap = "N/A", "N/A"

    return model, calibrator, features, auc, ap


# Load once at import time
try:
    model, calibrator, FEATURES, _auc, _ap = _load_model()
    _MODEL_LOADED = True
except Exception as _load_err:
    log.critical(f"Pipeline could not load model: {_load_err}")
    model = calibrator = None
    FEATURES = _DEFAULT_FEATURES
    _MODEL_LOADED = False


# ── Feature computation ───────────────────────────────────────────────────────
def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes only features that are MISSING from the DataFrame.
    If the CSV already has them (from prepare_merged_dataset.py),
    they are used as-is — no recomputation, no corruption.
    """
    df = df.copy()

    if "kWh electricity/day" in df.columns and "kWh_elec_per_day" not in df.columns:
        df["kWh_elec_per_day"] = df["kWh electricity/day"]

    if "TEMP_7DAY_AVG"  not in df.columns: df["TEMP_7DAY_AVG"]  = df["T2M"].rolling(7, min_periods=1).mean()
    if "SOLAR_7DAY_AVG" not in df.columns: df["SOLAR_7DAY_AVG"] = df["ALLSKY_SFC_SW_DWN"].rolling(7, min_periods=1).mean()

    if "Performance_Ratio" not in df.columns:
        irr = df["ALLSKY_SFC_SW_DWN"].clip(lower=1e-6)  # epsilon prevents div-by-zero & NaN cascade
        df["Performance_Ratio"] = (df["Daily_kWh"] / irr).fillna(0)

    pr = df["Performance_Ratio"]

    if "PR_3DAY_AVG"  not in df.columns: df["PR_3DAY_AVG"]  = pr.rolling(3,  min_periods=1).mean()
    if "PR_14DAY_AVG" not in df.columns: df["PR_14DAY_AVG"] = pr.rolling(14, min_periods=1).mean()
    if "PR_30DAY_AVG" not in df.columns: df["PR_30DAY_AVG"] = pr.rolling(30, min_periods=1).mean()
    if "PR_7DAY_STD"  not in df.columns: df["PR_7DAY_STD"]  = pr.rolling(7,  min_periods=2).std().fillna(0)
    if "PR_30DAY_STD" not in df.columns: df["PR_30DAY_STD"] = pr.rolling(30, min_periods=2).std().fillna(0)

    if "PR_Z_SCORE" not in df.columns:
        _EPS = 1e-8
        df["PR_Z_SCORE"] = ((pr - df["PR_30DAY_AVG"]) / df["PR_30DAY_STD"].clip(lower=_EPS)).fillna(0)

    if "PR_7DAY_TREND" not in df.columns:
        _x7 = np.arange(7, dtype=float)
        df["PR_7DAY_TREND"] = (
            pr.rolling(7, min_periods=7)
            .apply(lambda y: np.polyfit(_x7, y, 1)[0], raw=True)
            .fillna(0)
        )

    if "PR_LAG1"            not in df.columns: df["PR_LAG1"]            = pr.shift(1).bfill()
    if "PR_LAG3"            not in df.columns: df["PR_LAG3"]            = pr.shift(3).bfill()
    if "Daily_kWh_7DAY_AVG" not in df.columns: df["Daily_kWh_7DAY_AVG"] = df["Daily_kWh"].rolling(7, min_periods=1).mean()
    if "TEMP_ANOMALY"       not in df.columns: df["TEMP_ANOMALY"]       = (df["T2M"] - df["TEMP_7DAY_AVG"]).fillna(0)
    if "SOLAR_RATIO"        not in df.columns: df["SOLAR_RATIO"]        = (df["ALLSKY_SFC_SW_DWN"] / df["SOLAR_7DAY_AVG"].clip(lower=1e-8)).fillna(1)

    if "DOY_SIN" not in df.columns or "DOY_COS" not in df.columns:
        doy = df["DATE"].dt.dayofyear
        df["DOY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
        df["DOY_COS"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main inference entry point.
    Gracefully handles feature mismatches: fills missing features with 0
    and logs a warning instead of crashing the app.
    """
    if not _MODEL_LOADED:
        raise RuntimeError(
            "Model not loaded — check solar_ai.log for details, then re-run train_model.py."
        )

    df = _ensure_features(df)

    # ── Graceful feature mismatch handling ───────────────────────────────────
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Pipeline aborted — dataset is missing {len(missing)} required feature(s): {missing}.\n"
            "Re-run prepare_merged_dataset.py to regenerate the full feature set.\n"
            "Predictions with zero-filled features would be statistically invalid."
        )

    extra = [c for c in df.columns if c not in FEATURES and c not in ["DATE", "Anomaly"]]
    if extra:
        log.debug(f"Dataset has {len(extra)} extra columns not used by model — safely ignored.")

    X = df[FEATURES].ffill().bfill()

    raw_probs = model.predict_proba(X)[:, 1]

    # ── Probability smoothing: 7-day rolling average ──────────────────────────
    # Raw probabilities from the ensemble can be very sharp (near 0 or 1)
    # because the model was trained on clean labels. A 7-day rolling average
    # makes the probability curve look like a realistic continuous risk signal
    # rather than a binary classifier output — more appropriate for display.
    raw_series = pd.Series(raw_probs)
    smoothed_probs = raw_series.rolling(window=7, min_periods=1).mean().values

    df["Anomaly_Probability"]  = calibrator.predict(smoothed_probs) if calibrator else smoothed_probs
    df["Predicted"]            = model.predict(X)
    df["Adjusted_Probability"] = df["Anomaly_Probability"]

    ap = df["Anomaly_Probability"]
    log.info(f"Inference complete — {len(df):,} rows | prob min={ap.min():.3f}  mean={ap.mean():.3f}  max={ap.max():.3f}")

    return df
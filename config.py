"""
config.py — Central configuration for Solar AI
All paths, thresholds, and constants live here.
Never hardcode these values in other files.
"""
import os

# ── Base directory (project root, wherever this file lives) ─────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(*parts):
    return os.path.join(BASE_DIR, *parts)

# ── File Paths ───────────────────────────────────────────────────────────────
DATASET_PATH   = _path("solar_predictive_maintenance_dataset.csv")
MODEL_PATH     = _path("model.pkl")
DB_PATH        = _path("solar.db")
KAGGLE_PV_PATH = _path("kaggle_pv_daily_clean.csv")
NASA_PATH      = _path("nasa_power_daily_netherlands.csv")
LOG_PATH       = _path("solar_ai.log")

# ── Database ─────────────────────────────────────────────────────────────────
DB_TABLE          = "solar_data"
DB_IF_EXISTS      = "append"      # "append" preserves history; use "replace" to reset
DB_DATE_COL       = "DATE"        # used for deduplication on append

# ── Anomaly / Maintenance Thresholds ─────────────────────────────────────────
THRESHOLD_HIGH    = 0.70   # Anomaly_Probability ≥ this → MAINTENANCE REQUIRED
THRESHOLD_MEDIUM  = 0.40   # Anomaly_Probability ≥ this → INSPECTION ADVISED

# ── Health Index ─────────────────────────────────────────────────────────────
HEALTH_FLOOR      = 0.0    # True floor — allows real 0% for genuine failure days
HEALTH_SMOOTHING  = 7      # Rolling window (days) for smoothed health display

# ── Fault Classification Rules ───────────────────────────────────────────────
FAULT_WEATHER_IRRADIANCE = 2.0   # kWh/m²/day below this → WEATHER
FAULT_DUST_PR            = 0.6   # Performance Ratio below this → DUST / SOILING
FAULT_TEMP_CELSIUS       = 35.0  # Ambient temp above this → TEMPERATURE LOSS

# ── Revenue Model ─────────────────────────────────────────────────────────────
TARIFF_INR_PER_KWH = 10.0  # Illustrative tariff — update for real deployments

# ── Anomaly Label (training) ──────────────────────────────────────────────────
ANOMALY_PR_THRESHOLD = 0.75   # PR < (rolling_mean * this) → Anomaly label = 1
ANOMALY_ROLLING_DAYS = 30

# ── Forecast ──────────────────────────────────────────────────────────────────
FORECAST_DAYS    = 14
FORECAST_ALPHA   = 0.25   # Holt's level smoothing
FORECAST_BETA    = 0.08   # Holt's trend smoothing
FORECAST_WINDOW  = 60     # Days of history used as forecast context

# ── NASA POWER API ────────────────────────────────────────────────────────────
NASA_LAT    = 52.37
NASA_LON    = 4.90
NASA_START  = "20111026"
NASA_END    = "20221231"
NASA_PARAMS = "ALLSKY_SFC_SW_DWN,T2M"

# ── Probability Smoothing (display) ──────────────────────────────────────────
PROB_DISPLAY_SMOOTH = 7   # Rolling window to smooth probability curve in chart

# ── Configuration Validation ──────────────────────────────────────────────────
# Runs at import time — catches misconfiguration before it silently corrupts results
def _validate_config():
    assert 0 < THRESHOLD_MEDIUM < THRESHOLD_HIGH < 1.0, (
        f"Thresholds must satisfy 0 < THRESHOLD_MEDIUM ({THRESHOLD_MEDIUM}) "
        f"< THRESHOLD_HIGH ({THRESHOLD_HIGH}) < 1.0"
    )
    assert HEALTH_FLOOR >= 0.0, f"HEALTH_FLOOR must be ≥ 0, got {HEALTH_FLOOR}"
    assert FORECAST_DAYS > 0,   f"FORECAST_DAYS must be > 0, got {FORECAST_DAYS}"
    assert FORECAST_WINDOW > FORECAST_DAYS, (
        f"FORECAST_WINDOW ({FORECAST_WINDOW}) must exceed FORECAST_DAYS ({FORECAST_DAYS})"
    )
    assert 0 < FORECAST_ALPHA < 1, f"FORECAST_ALPHA must be in (0,1), got {FORECAST_ALPHA}"
    assert 0 < FORECAST_BETA  < 1, f"FORECAST_BETA must be in (0,1), got {FORECAST_BETA}"
    assert TARIFF_INR_PER_KWH > 0, f"TARIFF_INR_PER_KWH must be > 0"

_validate_config()
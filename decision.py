"""
decision.py — Post-inference decision logic for Solar AI
Applies maintenance thresholds, fault classification, and loss estimation.
All thresholds are imported from config.py — never hardcoded here.
"""
import numpy as np
import pandas as pd
from config import (
    THRESHOLD_HIGH, THRESHOLD_MEDIUM,
    HEALTH_FLOOR,
    FAULT_WEATHER_IRRADIANCE, FAULT_DUST_PR, FAULT_TEMP_CELSIUS,
    TARIFF_INR_PER_KWH,
)
from logger import get_logger

log = get_logger("solar_ai.decision")


def maintenance_status(p: float) -> str:
    if p >= THRESHOLD_HIGH:
        return "MAINTENANCE REQUIRED"
    elif p >= THRESHOLD_MEDIUM:
        return "INSPECTION ADVISED"
    return "HEALTHY"


def maintenance_priority(status: str) -> str:
    if status == "MAINTENANCE REQUIRED":
        return "HIGH"
    elif status == "INSPECTION ADVISED":
        return "MEDIUM"
    return "LOW"


def fault_type(row: pd.Series) -> str:
    """
    Rule-based fault classifier.
    Priority order: WEATHER → DUST/SOILING → TEMPERATURE LOSS → NORMAL
    Thresholds are configurable via config.py.
    """
    if row["ALLSKY_SFC_SW_DWN"] < FAULT_WEATHER_IRRADIANCE:
        return "WEATHER"
    elif row["Performance_Ratio"] < FAULT_DUST_PR:
        return "DUST / SOILING"
    elif row["T2M"] > FAULT_TEMP_CELSIUS:
        return "TEMPERATURE LOSS"
    return "NORMAL"


def _smooth_health(prob_series: pd.Series) -> pd.Series:
    """
    Applies a 7-day rolling mean to the raw (1-prob)*100 health index,
    then clips to [HEALTH_FLOOR, 100] to prevent unrealistic 0-drops.
    This makes the health curve look like gradual degradation rather
    than binary spikes — which is more realistic and visually cleaner.
    """
    raw = (1 - prob_series) * 100
    smoothed = raw.rolling(window=7, min_periods=1).mean()
    return smoothed.clip(lower=HEALTH_FLOOR, upper=100).round(2)


def apply_decision_logic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prob = (
        df["Adjusted_Probability"]
        if "Adjusted_Probability" in df.columns
        else df["Anomaly_Probability"]
    )

    df["Maintenance_Status"]   = prob.apply(maintenance_status)
    df["Maintenance_Priority"] = df["Maintenance_Status"].apply(maintenance_priority)

    # ── Smoothed Health Index (prevents unrealistic 0-drops) ────────────────
    df["Health_Index"]   = _smooth_health(prob)
    df["Severity_Score"] = (prob * 100).round(2)

    df["Fault_Type"] = df.apply(fault_type, axis=1)

    # ── Energy & Revenue Loss ────────────────────────────────────────────────
    df["Expected_kWh"] = (
        df["ALLSKY_SFC_SW_DWN"] *
        df["Performance_Ratio"].rolling(30, min_periods=1).mean()
    )

    df["Energy_Loss_kWh"] = (
        df["Expected_kWh"] * (1 - df["Health_Index"] / 100)
    ).clip(lower=0)

    df["Revenue_Loss_INR"] = (df["Energy_Loss_kWh"] * TARIFF_INR_PER_KWH).round(2)

    high  = (df["Maintenance_Priority"] == "HIGH").sum()
    med   = (df["Maintenance_Priority"] == "MEDIUM").sum()
    total = len(df)
    log.info(
        f"Decision logic applied — {total:,} rows | "
        f"HIGH: {high:,} | MEDIUM: {med:,} | "
        f"Health min: {df['Health_Index'].min():.1f}%  mean: {df['Health_Index'].mean():.1f}%"
    )

    return df
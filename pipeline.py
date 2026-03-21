import pandas as pd
import joblib

print("⚙️ Loading model...")

model = joblib.load("model.pkl")

FEATURES = [
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "TEMP_7DAY_AVG",
    "SOLAR_7DAY_AVG",
    "Performance_Ratio"
]

def run_pipeline(df):
    df = df.copy()

    X = df[FEATURES].ffill().bfill()

    # =============================
    # MODEL PREDICTION
    # =============================
    df["Anomaly_Probability"] = model.predict_proba(X)[:, 1]
    df["Predicted"] = model.predict(X)

    # =============================
    # 🔥 ADAPTATION LAYER (KEY UPGRADE)
    # =============================
    baseline_pr = df["Performance_Ratio"].tail(7).mean()

    # Avoid division issues
    baseline_pr = baseline_pr if baseline_pr != 0 else 1

    df["Adaptive_Score"] = df["Performance_Ratio"] / baseline_pr

    # Adjust anomaly probability
    df["Adjusted_Probability"] = (
        df["Anomaly_Probability"] * (1 / df["Adaptive_Score"])
    ).clip(0, 1)

    return df
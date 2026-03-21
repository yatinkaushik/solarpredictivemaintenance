def maintenance_status(p):
    if p >= 0.7:
        return "MAINTENANCE REQUIRED"
    elif p >= 0.4:
        return "INSPECTION ADVISED"
    else:
        return "HEALTHY"


def maintenance_priority(status):
    if status == "MAINTENANCE REQUIRED":
        return "HIGH"
    elif status == "INSPECTION ADVISED":
        return "MEDIUM"
    else:
        return "LOW"


def fault_type(row):
    if row["ALLSKY_SFC_SW_DWN"] < 2:
        return "WEATHER"
    elif row["Performance_Ratio"] < 0.6:
        return "DUST / SOILING"
    elif row["T2M"] > 35:
        return "TEMPERATURE LOSS"
    else:
        return "NORMAL"


def apply_decision_logic(df):
    df = df.copy()

    # =============================
    # 🔥 USE ADAPTIVE PROBABILITY
    # =============================
    prob = df["Adjusted_Probability"] if "Adjusted_Probability" in df else df["Anomaly_Probability"]

    df["Maintenance_Status"] = prob.apply(maintenance_status)
    df["Maintenance_Priority"] = df["Maintenance_Status"].apply(maintenance_priority)

    df["Health_Index"] = ((1 - prob) * 100).round(2)
    df["Severity_Score"] = (prob * 100).round(2)

    df["Fault_Type"] = df.apply(fault_type, axis=1)

    # =============================
    # ENERGY + REVENUE LOSS
    # =============================
    df["Expected_kWh"] = (
        df["ALLSKY_SFC_SW_DWN"] *
        df["Performance_Ratio"].rolling(30, min_periods=1).mean()
    )

    df["Energy_Loss_kWh"] = (
        df["Expected_kWh"] * (1 - df["Health_Index"] / 100)
    ).clip(lower=0)

    df["Revenue_Loss_INR"] = (df["Energy_Loss_kWh"] * 10).round(2)

    return df
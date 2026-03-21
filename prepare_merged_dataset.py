import pandas as pd
import numpy as np

print("Loading cleaned PV data...")

# =====================================================
# 1. LOAD PV DATA
# =====================================================
pv_df = pd.read_csv("kaggle_pv_daily_clean.csv")

pv_df["date"] = pd.to_datetime(pv_df["date"])
pv_df.rename(columns={"date": "DATE"}, inplace=True)

print("PV shape:", pv_df.shape)

# =====================================================
# 2. LOAD NASA POWER DATA
# =====================================================
print("\nLoading NASA POWER data...")

nasa_df = pd.read_csv("nasa_power_daily_netherlands.csv")

nasa_df["DATE"] = pd.to_datetime(nasa_df["DATE"])

print("NASA shape:", nasa_df.shape)

# =====================================================
# 3. MERGE DATASETS
# =====================================================
merged_df = pd.merge(
    pv_df,
    nasa_df,
    on="DATE",
    how="inner"
)

print("\nMerged shape:", merged_df.shape)

if merged_df.empty:
    raise Exception("❌ Merge failed — date mismatch")

# =====================================================
# 4. CREATE ROLLING FEATURES (VERY IMPORTANT)
# =====================================================
merged_df["TEMP_7DAY_AVG"] = (
    merged_df["T2M"].rolling(window=7, min_periods=1).mean()
)

merged_df["SOLAR_7DAY_AVG"] = (
    merged_df["ALLSKY_SFC_SW_DWN"].rolling(window=7, min_periods=1).mean()
)

# =====================================================
# WEATHER-NORMALIZED PERFORMANCE MODEL
# =====================================================

# Estimate expected generation from irradiance
# (simple linear proxy — acceptable for academic work)

irradiance = merged_df["ALLSKY_SFC_SW_DWN"]

# Avoid division issues
irradiance_safe = irradiance.replace(0, np.nan)

# Performance ratio (actual vs weather-expected)
merged_df["Performance_Ratio"] = (
    merged_df["Daily_kWh"] / irradiance_safe
)

merged_df["Performance_Ratio"] = merged_df["Performance_Ratio"].fillna(0)

# Rolling expected performance
expected_pr = merged_df["Performance_Ratio"].rolling(
    window=30, min_periods=1
).mean()

# =====================================================
# SMART ANOMALY LABEL
# =====================================================

merged_df["Anomaly"] = np.where(
    merged_df["Performance_Ratio"] < expected_pr * 0.75,
    1,
    0
)

print("\nWeather-normalized anomaly distribution:")
print(merged_df["Anomaly"].value_counts())

print("\nAnomaly distribution:")
print(merged_df["Anomaly"].value_counts())

# =====================================================
# 6. SAVE FINAL MODEL DATASET
# =====================================================
output_file = "solar_predictive_maintenance_dataset.csv"
merged_df.to_csv(output_file, index=False)

print("\n✅ Final merged dataset saved as:", output_file)

print("\n🎯 READY for AI model training")
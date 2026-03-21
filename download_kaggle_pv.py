import kagglehub
import os
import pandas as pd

print("Downloading Kaggle solar PV dataset...")

# =====================================================
# 1. DOWNLOAD DATASET
# =====================================================
path = kagglehub.dataset_download("fvcoppen/solarpanelspower")
print("✅ Dataset downloaded to:", path)

# =====================================================
# 2. LOAD MAIN CSV
# =====================================================
csv_path = os.path.join(path, "PV_Elec_Gas3.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError("PV_Elec_Gas3.csv not found in dataset.")

print("\nLoading:", csv_path)

# IMPORTANT: file is comma-separated (not semicolon)
df = pd.read_csv(csv_path)

print("\nRaw columns:", df.columns.tolist())
print(df.head())

# =====================================================
# 3. CLEAN COLUMN NAMES
# =====================================================
df.columns = df.columns.str.strip()

# =====================================================
# 4. CONVERT DATE
# =====================================================
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

# Remove invalid dates
df = df.dropna(subset=["date"])

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

# =====================================================
# 5. CREATE DAILY SOLAR GENERATION
# (CRITICAL FOR PREDICTIVE MAINTENANCE)
# =====================================================
df["Daily_kWh"] = df["Cumulative_solar_power"].diff()

# First row fix
df["Daily_kWh"] = df["Daily_kWh"].fillna(0)

# Remove negative spikes (sensor resets)
df.loc[df["Daily_kWh"] < 0, "Daily_kWh"] = 0

# =====================================================
# 6. BASIC DATA QUALITY CHECK
# =====================================================
print("\n=== After Daily Conversion ===")
print(df[["date", "Cumulative_solar_power", "Daily_kWh"]].head())

print("\nDataset shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())

# =====================================================
# 7. SAVE CLEAN FILE
# =====================================================
output_file = "kaggle_pv_daily_clean.csv"
df.to_csv(output_file, index=False)

print("\n✅ Clean PV file saved as:", output_file)

print("\n🎯 READY for NASA POWER merge")
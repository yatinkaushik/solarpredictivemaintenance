import requests
import pandas as pd

# =========================================
# NETHERLANDS LOCATION
# =========================================
LAT = 52.37
LON = 4.90

# Match Kaggle dataset duration
START = "20111026"
END = "20221231"

PARAMETERS = "ALLSKY_SFC_SW_DWN,T2M"

# =========================================
# NASA POWER API URL
# =========================================
url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters={PARAMETERS}"
    "&community=RE"
    f"&longitude={LON}"
    f"&latitude={LAT}"
    f"&start={START}"
    f"&end={END}"
    "&format=JSON"
)

print("Downloading NASA POWER data for Netherlands...")

response = requests.get(url, timeout=60)
response.raise_for_status()
data = response.json()

# =========================================
# PARSE DATA
# =========================================
records = data["properties"]["parameter"]
df = pd.DataFrame(records)

df.index = pd.to_datetime(df.index)
df.reset_index(inplace=True)
df.rename(columns={"index": "DATE"}, inplace=True)

# =========================================
# SAVE
# =========================================
output_file = "nasa_power_daily_netherlands.csv"
df.to_csv(output_file, index=False)

print("✅ Saved:", output_file)
print(df.head())
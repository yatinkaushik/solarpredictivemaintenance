import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

print("🚀 Training model...")

# Load dataset (same folder)
df = pd.read_csv("solar_predictive_maintenance_dataset.csv")

df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

# Features
features = [
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "TEMP_7DAY_AVG",
    "SOLAR_7DAY_AVG",
    "Performance_Ratio"
]

X = df[features].ffill().bfill()
y = df["Anomaly"]

# Time-based split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# Model
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Save model (same folder)
joblib.dump(model, "model.pkl")

print("✅ Model saved as model.pkl")
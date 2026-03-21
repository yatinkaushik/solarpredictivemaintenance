import streamlit as st
import pandas as pd
from pipeline import run_pipeline
from decision import apply_decision_logic
from db import save_to_db


st.set_page_config(page_title="Solar AI Dashboard", layout="wide")

st.caption("AI-powered decision intelligence system for solar PV maintenance using satellite and operational data.")

st.title("☀️ AI-Based Solar Predictive Maintenance System")

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("solar_predictive_maintenance_dataset.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE")

# =============================
# RUN SYSTEM
# =============================
df = run_pipeline(df)
df = apply_decision_logic(df)

# =============================
# DEFINE LATEST (FIX HERE)
# =============================
latest = df.iloc[-1]

save_to_db(df)

# =============================
# RECOMMENDED ACTION (NOW SAFE)
# =============================
st.subheader("🛠️ Recommended Action")

if latest["Maintenance_Priority"] == "HIGH":
    st.error("🚨 Immediate maintenance required. Dispatch team now.")
elif latest["Maintenance_Priority"] == "MEDIUM":
    st.warning("⚠️ Schedule preventive maintenance soon.")
else:
    st.success("✅ No action required. System stable.")
# =============================
# DATE FILTER
# =============================
st.sidebar.header("Filter Data")

start_date = st.sidebar.date_input("Start Date", df["DATE"].min())
end_date = st.sidebar.date_input("End Date", df["DATE"].max())

df = df[(df["DATE"] >= pd.to_datetime(start_date)) &
        (df["DATE"] <= pd.to_datetime(end_date))]

# =============================
# TOP METRICS
# =============================
latest = df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Health Index", f"{latest['Health_Index']:.2f}")
col2.metric("Anomaly Prob.", f"{latest['Anomaly_Probability']:.2f}")
col3.metric("Status", latest["Maintenance_Status"])
col4.metric("Revenue Loss ₹", f"{latest['Revenue_Loss_INR']:.2f}")

# =============================
# EXECUTIVE SUMMARY
# =============================
st.subheader("📊 Executive Summary")

critical = (df["Maintenance_Priority"] == "HIGH").sum()
warning = (df["Maintenance_Priority"] == "MEDIUM").sum()
total_loss = df["Revenue_Loss_INR"].sum()

if critical > 0:
    st.error(f"🚨 {critical} critical issues detected. Immediate action required.")
elif warning > 0:
    st.warning(f"⚠️ {warning} warnings detected. Preventive maintenance advised.")
else:
    st.success("✅ System operating normally.")

st.info(f"💰 Total Estimated Revenue Loss: ₹{total_loss:.2f}")

# =============================
# HEALTH BAR
# =============================
st.subheader("🔋 System Health Level")

health = latest["Health_Index"]

st.progress(int(health))

if health < 50:
    st.error("🚨 System Health Critical")
elif health < 75:
    st.warning("⚠️ System Performance Degrading")
else:
    st.success("✅ System Healthy")

# =============================
# TREND CHART
# =============================
st.subheader("📈 Performance Ratio Trend")
st.line_chart(df.set_index("DATE")["Performance_Ratio"])

# =============================
# ENERGY LOSS CHART
# =============================
st.subheader("⚡ Energy Loss Trend")
st.line_chart(df.set_index("DATE")["Energy_Loss_kWh"])

# =============================
# SMART ALERT SUMMARY
# =============================
st.subheader("🚨 Smart Alerts Summary")

critical_days = df[df["Maintenance_Priority"] == "HIGH"].tail(5)

if len(critical_days) > 0:
    st.error("⚠️ Critical maintenance required in recent days")
else:
    st.success("✅ No critical issues detected")

st.dataframe(critical_days[[
    "DATE",
    "Maintenance_Status",
    "Fault_Type",
    "Revenue_Loss_INR"
]])

# =============================
# TOP LOSS DAYS
# =============================
st.subheader("💸 Top Revenue Loss Days")

top_loss = df.sort_values("Revenue_Loss_INR", ascending=False).head(5)

st.dataframe(top_loss[[
    "DATE",
    "Revenue_Loss_INR",
    "Maintenance_Status",
    "Fault_Type"
]])

# =============================
# TREND WARNING
# =============================
st.subheader("📉 System Trend Analysis")

recent_health = df["Health_Index"].tail(7).mean()
overall_health = df["Health_Index"].mean()

if recent_health < overall_health * 0.9:
    st.warning("⚠️ System performance declining in recent days")
else:
    st.success("✅ System performance stable")

# =============================
# FUTURE PREDICTION (SIMPLE)
# =============================
st.subheader("🔮 Future Risk Prediction (Next 7 Days)")

recent = df.tail(7)

future_risk = recent["Anomaly_Probability"].mean()

if future_risk > 0.6:
    st.error("🚨 High risk of future failure in coming days")
elif future_risk > 0.4:
    st.warning("⚠️ Moderate risk detected")
else:
    st.success("✅ System expected to remain stable")

st.metric("Predicted Risk Score", f"{future_risk:.2f}")

# =============================
# FAULT DISTRIBUTION
# =============================
st.subheader("🧠 Fault Distribution")

fault_counts = df["Fault_Type"].value_counts()

st.bar_chart(fault_counts)

# =============================
# BUSINESS IMPACT SUMMARY
# =============================
st.subheader("💼 Business Impact Analysis")

total_loss = df["Revenue_Loss_INR"].sum()
avg_loss = df["Revenue_Loss_INR"].mean()
max_loss = df["Revenue_Loss_INR"].max()

col1, col2, col3 = st.columns(3)

col1.metric("Total Loss ₹", f"{total_loss:.2f}")
col2.metric("Avg Daily Loss ₹", f"{avg_loss:.2f}")
col3.metric("Worst Day Loss ₹", f"{max_loss:.2f}")
# ☀️ Solar Predictive Maintenance AI

🚨 Detects solar panel failures before they happen
💰 Estimates revenue loss in real-time
📊 End-to-end ML system with dashboard + decision engine

---

## 🚀 Overview

Solar power systems often lose efficiency due to unnoticed faults like dust accumulation, weather impact, or temperature loss.

This project builds an **AI-powered predictive maintenance system** that:

* Detects anomalies in solar performance
* Predicts failure probability
* Classifies fault types
* Estimates financial loss
* Recommends maintenance actions

👉 Designed to simulate a **real-world production ML system**, not just a model.

---

## 🎯 Key Features

### 🔍 Anomaly Detection

* Predicts probability of system failure
* Uses ensemble ML model with calibrated outputs
* Time-series aware (no data leakage)

---

### 🧠 Feature Engineering (Core Strength)

* Performance Ratio (PR)
* PR Z-score (main anomaly signal)
* Rolling averages & trends
* Lag features (temporal memory)
* Seasonal encoding (sin/cos)

---

### ⚙️ Decision Engine

Transforms predictions into business actions:

* **Maintenance Status**

  * Healthy
  * Inspection Advised
  * Maintenance Required

* **Fault Detection**

  * Weather-related
  * Dust / Soiling
  * Temperature loss

---

### 💰 Business Impact

* Energy loss (kWh)
* Revenue loss (₹)
* System health index (0–100%)

---

### 🔮 Forecasting

* 14-day risk prediction
* Confidence intervals
* Trend direction (improving / degrading)

---

### 📊 Dashboard (Streamlit)

👉 Interactive UI with:

* KPI cards (health, risk, loss)
* Performance trends
* Anomaly probability graph
* Forecast visualization
* Fault distribution
* Maintenance logs

---

## 🏗️ Architecture

```id="arch2"
Data Sources
   ├── Kaggle Solar Dataset
   └── NASA POWER API
            ↓
Data Cleaning & Merge
            ↓
Feature Engineering (19 features)
            ↓
Model Training (Ensemble + Calibration)
            ↓
Inference Pipeline
            ↓
Decision Engine
            ↓
SQLite Database
            ↓
Streamlit Dashboard
```

---

## 📁 Project Structure

```id="struct2"
.
├── app.py                  # Streamlit dashboard
├── pipeline.py             # ML inference pipeline
├── decision.py             # Maintenance logic
├── train_model.py          # Model training
├── prepare_merged_dataset.py
├── db.py                   # Database logic
├── config.py               # Config & thresholds
├── logger.py               # Logging
├── requirements.txt
├── model.pkl
└── dataset.csv
```

---

## ⚙️ Installation

```bash id="inst1"
git clone https://github.com/your-username/solarpredictivemaintenance.git
cd solarpredictivemaintenance
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash id="run1"
streamlit run app.py
```

---

## 📊 Model Details

* **Type:** Ensemble (Voting Classifier)
* **Models Used:**

  * Random Forest
  * Gradient Boosting
  * Extra Trees
* **Calibration:** Isotonic Regression
* **Validation:** Time-series split

---

## 📈 Key Insight

The model primarily learns:

> **Deviation from normal performance (PR Z-score)**

This makes it highly effective at detecting anomalies even under changing weather conditions.

---

## ⚠️ Limitations

* Uses synthetic anomaly labels (rule-based)
* Fault classification is heuristic (not ML-based)
* Batch processing (not real-time)
* Some feature redundancy

---

## 🚀 Future Improvements

* Real anomaly data / unsupervised detection
* ML-based fault classification
* SHAP explainability
* Real-time streaming pipeline
* Deep learning (LSTM / transformers)

---

## 🧪 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Plotly
* SQLite

---

## 📌 Author

**Yatin Kaushik**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐

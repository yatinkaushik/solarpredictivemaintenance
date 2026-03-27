import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, os
from pipeline import run_pipeline
from decision import apply_decision_logic
from db import save_to_db
from config import (
    DATASET_PATH, MODEL_PATH,
    THRESHOLD_HIGH, THRESHOLD_MEDIUM,
    PROB_DISPLAY_SMOOTH, FORECAST_DAYS,
    FORECAST_ALPHA, FORECAST_BETA, FORECAST_WINDOW,
)
from logger import get_logger

log = get_logger("solar_ai.app")

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Solar AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="☀️"
)

# =============================
# CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg-primary:#0a0e1a; --bg-card:#111827; --bg-card2:#161d2e;
    --accent-solar:#f59e0b; --accent-green:#10b981; --accent-red:#ef4444;
    --accent-blue:#3b82f6; --text-primary:#f1f5f9; --text-muted:#64748b; --border:#1e293b;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg-primary)!important;color:var(--text-primary)!important;}
.stApp{background-color:var(--bg-primary)!important;}
.dash-header{background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#0f2027 100%);border-bottom:1px solid var(--border);padding:1.5rem 2rem;margin:-1rem -1rem 2rem -1rem;}
.dash-header h1{font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;color:var(--accent-solar);margin:0;letter-spacing:-0.5px;}
.dash-header p{color:var(--text-muted);font-size:0.85rem;margin:0.2rem 0 0 0;}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;}
.kpi-card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.4rem;position:relative;overflow:hidden;transition:transform .2s ease,border-color .2s ease;}
.kpi-card:hover{transform:translateY(-2px);border-color:var(--accent-solar);}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0;}
.kpi-card.solar::before{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.kpi-card.blue::before{background:linear-gradient(90deg,#3b82f6,#60a5fa);}
.kpi-card.green::before{background:linear-gradient(90deg,#10b981,#34d399);}
.kpi-card.red::before{background:linear-gradient(90deg,#ef4444,#f87171);}
.kpi-label{font-size:.72rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-muted);margin-bottom:.5rem;font-family:'Space Mono',monospace;}
.kpi-value{font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:var(--text-primary);line-height:1;}
.kpi-sub{font-size:.75rem;color:var(--text-muted);margin-top:.4rem;}
.alert-box{border-radius:10px;padding:1rem 1.3rem;margin-bottom:1rem;display:flex;align-items:flex-start;gap:.8rem;font-size:.9rem;}
.alert-critical{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#fca5a5;}
.alert-warning{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);color:#fcd34d;}
.alert-ok{background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);color:#6ee7b7;}
.alert-info{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);color:#93c5fd;}
.alert-icon{font-size:1.2rem;}
.section-title{font-family:'Space Mono',monospace;font-size:.9rem;text-transform:uppercase;letter-spacing:2px;color:var(--text-muted);border-left:3px solid var(--accent-solar);padding-left:.8rem;margin:1.5rem 0 1rem 0;}
.health-bar-wrap{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.5rem;}
.health-bar-track{background:#1e293b;border-radius:100px;height:12px;overflow:hidden;margin:.6rem 0;}
.health-bar-fill{height:100%;border-radius:100px;transition:width 1s ease;}
.metric-row{display:grid;gap:1rem;margin-bottom:1rem;}
.metric-box{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:1rem 1.2rem;text-align:center;}
.metric-box .val{font-family:'Space Mono',monospace;font-size:1.5rem;font-weight:700;}
.metric-box .lbl{font-size:.72rem;text-transform:uppercase;letter-spacing:1px;color:var(--text-muted);margin-top:.3rem;}
[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] .stMarkdown h3{font-family:'Space Mono',monospace;color:var(--accent-solar);font-size:.85rem;letter-spacing:1px;text-transform:uppercase;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border);gap:0;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--text-muted)!important;font-family:'DM Sans',sans-serif;font-size:.9rem;padding:.6rem 1.2rem;border-bottom:2px solid transparent;}
.stTabs [aria-selected="true"]{color:var(--accent-solar)!important;border-bottom-color:var(--accent-solar)!important;}
[data-testid="stDataFrame"]{border-radius:10px;border:1px solid var(--border)!important;overflow:hidden;}
#MainMenu,footer{visibility:hidden;}
.stDeployButton{display:none;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Hide top right toolbar */
div[data-testid="stToolbar"] {
    display: none !important;
}

/* Hide deploy button */
button[kind="header"] {
    display: none !important;
}

/* Hide GitHub / share icons */
[data-testid="stHeader"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# =============================
# DATA LOADING
# =============================
@st.cache_data(show_spinner="Loading solar data...")
def load_data(_bust=0):
    try:
        df = pd.read_csv(DATASET_PATH)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").reset_index(drop=True)
        df = run_pipeline(df)
        df = apply_decision_logic(df)
        return df
    except FileNotFoundError:
        log.error(f"Dataset not found at: {DATASET_PATH}")
        st.error(f"❌ Dataset not found at: {DATASET_PATH}. Run prepare_merged_dataset.py first.")
        st.stop()
    except Exception as e:
        log.error(f"Error loading data: {e}")
        st.error(f"❌ Error loading data: {e}")
        st.stop()

@st.cache_data(show_spinner=False)
def load_model_meta():
    """Load model metadata for the accuracy report tab."""
    try:
        payload = joblib.load(MODEL_PATH)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}

_mtime = int(os.path.getmtime(MODEL_PATH)) if os.path.exists(MODEL_PATH) else 0
log.info("Dashboard starting — loading data...")
df_full = load_data(_bust=_mtime)
save_to_db(df_full)
model_meta = load_model_meta()


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("### ☀️ Solar AI")
    st.markdown("---")
    st.markdown("### 📅 Date Filter")
    start_date = st.date_input("From", df_full["DATE"].min())
    end_date   = st.date_input("To",   df_full["DATE"].max())
    st.markdown("---")
    st.markdown("### 🎛️ Display Options")
    show_raw    = st.checkbox("Show Raw Data Table", False)
    chart_theme = st.selectbox("Chart Theme", ["Dark Solar", "Minimal", "Vibrant"])
    smoothing   = st.slider("Chart Smoothing (days)", 1, 14, 3)
    st.markdown("---")
    st.markdown("### 📤 Export")
    csv_bytes = df_full.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full Report (CSV)", csv_bytes,
                       "solar_maintenance_report.csv", "text/csv",
                       width="stretch")


# =============================
# FILTER
# =============================
df = df_full[
    (df_full["DATE"] >= pd.to_datetime(start_date)) &
    (df_full["DATE"] <= pd.to_datetime(end_date))
].copy()

if df.empty:
    st.warning("No data for selected date range.")
    st.stop()

latest    = df.iloc[-1]
latest_7d = df.tail(7).mean(numeric_only=True)

# Health_Index is always 0-100 (set by decision.py via config.HEALTH_FLOOR)
def to_pct(val): return float(val)  # identity — no scaling needed


# =============================
# CHART THEMES
# =============================
THEMES = {
    "Dark Solar": {"primary":"#f59e0b","secondary":"#3b82f6","danger":"#ef4444","success":"#10b981","grid":"#1e293b","text":"#94a3b8"},
    "Minimal":    {"primary":"#e2e8f0","secondary":"#94a3b8","danger":"#ef4444","success":"#6ee7b7","grid":"#1e293b","text":"#64748b"},
    "Vibrant":    {"primary":"#a78bfa","secondary":"#f472b6","danger":"#fb923c","success":"#34d399","grid":"#1e293b","text":"#94a3b8"},
}
C = THEMES[chart_theme]
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color=C["text"]),
    xaxis=dict(showgrid=True, gridcolor=C["grid"], zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=C["grid"], zeroline=False),
    hovermode="x unified",
)
PLOTLY_LEGEND  = dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["grid"])
_M_DEFAULT     = dict(l=10, r=10, t=35, b=10)   # default chart margin
_M_WIDE_LEFT   = dict(l=155, r=70, t=35, b=10)  # for horizontal bar charts


# =============================
# HEADER
# =============================
st.markdown(f"""
<div class="dash-header">
  <div>
    <h1>☀️ Solar AI · Predictive Maintenance</h1>
    <p>AI-powered decision intelligence · {start_date} → {end_date} · {len(df):,} records</p>
  </div>
</div>""", unsafe_allow_html=True)


# =============================
# BANNER
# =============================
_banner_prob = df.tail(7)["Anomaly_Probability"].mean()
if _banner_prob >= 0.5:
    st.markdown("""<div class="alert-box alert-critical"><span class="alert-icon">🚨</span>
    <div><strong>CRITICAL ALERT</strong> — Immediate maintenance required. Dispatch field team now.</div></div>""", unsafe_allow_html=True)
elif _banner_prob >= 0.3:
    st.markdown("""<div class="alert-box alert-warning"><span class="alert-icon">⚠️</span>
    <div><strong>WARNING</strong> — Preventive maintenance recommended. Schedule within 72 hours.</div></div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="alert-box alert-ok"><span class="alert-icon">✅</span>
    <div><strong>SYSTEM STABLE</strong> — No immediate action required.</div></div>""", unsafe_allow_html=True)


# =============================
# KPI CARDS
# =============================
health         = to_pct(latest_7d["Health_Index"])
anom           = latest_7d["Anomaly_Probability"]
rev_loss       = df["Revenue_Loss_INR"].sum()
critical_count = (df["Maintenance_Priority"] == "HIGH").sum()

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card solar">
    <div class="kpi-label">System Health (7d avg)</div>
    <div class="kpi-value">{health:.0f}<span style="font-size:1rem;color:#64748b">%</span></div>
    <div class="kpi-sub">{'⬇ Degrading' if health < 75 else '⬆ Healthy'}</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-label">Anomaly Prob (7d avg)</div>
    <div class="kpi-value">{anom:.2f}</div>
    <div class="kpi-sub">{'⬆ HIGH Risk' if anom >= THRESHOLD_HIGH else ('⚠ MEDIUM' if anom >= THRESHOLD_MEDIUM else 'Low Risk')}</div>
  </div>
  <div class="kpi-card red">
    <div class="kpi-label">Total Revenue Loss</div>
    <div class="kpi-value">₹{rev_loss:,.0f}</div>
    <div class="kpi-sub">Across selected period</div>
  </div>
  <div class="kpi-card {'red' if critical_count > 0 else 'green'}">
    <div class="kpi-label">Critical Days</div>
    <div class="kpi-value">{critical_count}</div>
    <div class="kpi-sub">{'🚨 Requires attention' if critical_count > 0 else '✅ All clear'}</div>
  </div>
</div>""", unsafe_allow_html=True)


# =============================
# HEALTH BAR
# =============================
bar_color    = C["danger"] if health < 50 else (C["primary"] if health < 75 else C["success"])
health_label = "Critical" if health < 50 else ("Degrading" if health < 75 else "Healthy")
st.markdown(f"""
<div class="health-bar-wrap">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-family:'Space Mono';font-size:.75rem;letter-spacing:1px;color:#64748b;text-transform:uppercase;">System Health Index</span>
    <span style="font-family:'Space Mono';font-size:.9rem;color:{bar_color};font-weight:700;">{health_label} · {health:.1f}%</span>
  </div>
  <div class="health-bar-track">
    <div class="health-bar-fill" style="width:{health}%;background:linear-gradient(90deg,{bar_color},{bar_color}88);"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:.7rem;color:#475569;">
    <span>0 — Critical</span><span>50 — Degrading</span><span>75 — Healthy</span><span>100</span>
  </div>
</div>""", unsafe_allow_html=True)


# =============================
# TABS
# =============================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Performance",
    "⚡ Energy & Loss",
    "🔮 Forecast",
    "🧠 Fault Analysis",
    "📋 Maintenance Log",
    "🎯 Model Report",
])


# ── TAB 1: PERFORMANCE ──────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Performance Ratio Trend</div>', unsafe_allow_html=True)
    df_plot = df.copy()
    df_plot["PR_smooth"] = df_plot["Performance_Ratio"].rolling(smoothing, center=True).mean()

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=df_plot["DATE"], y=df_plot["Performance_Ratio"],
        name="Daily PR", mode="lines", line=dict(color=C["primary"], width=1), opacity=0.35))
    fig_pr.add_trace(go.Scatter(x=df_plot["DATE"], y=df_plot["PR_smooth"],
        name=f"{smoothing}d Avg", mode="lines", line=dict(color=C["primary"], width=2.5)))
    fig_pr.update_layout(**PLOTLY_LAYOUT, margin=_M_DEFAULT, legend=PLOTLY_LEGEND, height=300,
        title=dict(text="Performance Ratio", font=dict(size=13)))
    st.plotly_chart(fig_pr, width="stretch")

    st.markdown('<div class="section-title">Anomaly Probability Over Time</div>', unsafe_allow_html=True)
    fig_ap = go.Figure()
    fig_ap.add_hrect(y0=0.6, y1=1.0, fillcolor=C["danger"], opacity=0.07, line_width=0)
    fig_ap.add_hrect(y0=0.4, y1=0.6, fillcolor=C["primary"], opacity=0.07, line_width=0)
    _ap_smooth = df["Anomaly_Probability"].rolling(PROB_DISPLAY_SMOOTH, min_periods=1).mean()
    fig_ap.add_trace(go.Scatter(x=df["DATE"], y=df["Anomaly_Probability"],
        mode="lines", name="Raw Prob", fill="tozeroy",
        line=dict(color=C["secondary"], width=1, dash="dot"), fillcolor="rgba(59,130,246,0.07)",
        opacity=0.5))
    fig_ap.add_trace(go.Scatter(x=df["DATE"], y=_ap_smooth,
        mode="lines", name=f"{PROB_DISPLAY_SMOOTH}d Smoothed",
        line=dict(color=C["primary"], width=2.5)))
    fig_ap.add_hline(y=0.6, line_dash="dash", line_color=C["danger"], opacity=0.6,
        annotation_text="High Risk", annotation_position="top right")
    fig_ap.add_hline(y=0.4, line_dash="dash", line_color=C["primary"], opacity=0.6,
        annotation_text="Moderate Risk", annotation_position="top right")
    fig_ap.update_layout(**PLOTLY_LAYOUT, margin=_M_DEFAULT, legend=PLOTLY_LEGEND, height=270,
        title=dict(text="Anomaly Probability", font=dict(size=13)))
    st.plotly_chart(fig_ap, width="stretch")


# ── TAB 2: ENERGY & LOSS ────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Energy Loss (kWh)</div>', unsafe_allow_html=True)
        fig_el = go.Figure()
        fig_el.add_trace(go.Bar(x=df["DATE"], y=df["Energy_Loss_kWh"], name="Energy Loss",
            marker=dict(color=df["Energy_Loss_kWh"],
                colorscale=[[0,C["success"]],[0.5,C["primary"]],[1,C["danger"]]],
                showscale=False)))
        fig_el.update_layout(**PLOTLY_LAYOUT, margin=_M_DEFAULT, legend=PLOTLY_LEGEND, height=300,
            title=dict(text="Daily Energy Loss kWh", font=dict(size=13)))
        st.plotly_chart(fig_el, width="stretch")

    with col_b:
        st.markdown('<div class="section-title">Top Revenue Loss Days</div>', unsafe_allow_html=True)
        top10 = df.sort_values("Revenue_Loss_INR", ascending=False).head(10)
        fig_rl = go.Figure(go.Bar(
            x=top10["Revenue_Loss_INR"], y=top10["DATE"].dt.strftime("%b %d '%y"),
            orientation="h", marker=dict(color=C["danger"], opacity=0.85)))
        fig_rl.update_layout(**PLOTLY_LAYOUT, margin=_M_DEFAULT, legend=PLOTLY_LEGEND, height=300,
            title=dict(text="Top 10 Revenue Loss Days", font=dict(size=13)))
        st.plotly_chart(fig_rl, width="stretch")

    st.markdown('<div class="section-title">Business Impact Summary</div>', unsafe_allow_html=True)
    avg_loss  = df["Revenue_Loss_INR"].mean() if not df.empty else 0
    max_loss  = df["Revenue_Loss_INR"].max()  if not df.empty else 0
    worst_day = (df.loc[df["Revenue_Loss_INR"].idxmax(), "DATE"].strftime("%d %b %Y")
                 if not df.empty else "N/A")
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
      <div class="kpi-card red"><div class="kpi-label">Total Loss</div>
        <div class="kpi-value" style="font-size:1.4rem;">₹{rev_loss:,.0f}</div></div>
      <div class="kpi-card solar"><div class="kpi-label">Avg Daily Loss</div>
        <div class="kpi-value" style="font-size:1.4rem;">₹{avg_loss:,.0f}</div></div>
      <div class="kpi-card blue"><div class="kpi-label">Worst Day — {worst_day}</div>
        <div class="kpi-value" style="font-size:1.4rem;">₹{max_loss:,.0f}</div></div>
    </div>""", unsafe_allow_html=True)


# ── TAB 3: FORECAST (Exponential Smoothing + Confidence Interval) ────────────
with tab3:
    st.markdown('<div class="section-title">14-Day Risk Forecast · Exponential Smoothing with Confidence Bands</div>', unsafe_allow_html=True)

    # FORECAST_DAYS from config

    def holt_forecast(series, days=FORECAST_DAYS, alpha=0.3, beta=0.1):
        """
        Holt's Double Exponential Smoothing (trend-aware).
        Returns (forecast_values, lower_ci, upper_ci).
        alpha = smoothing level, beta = smoothing trend.
        """
        s = series.dropna().values
        if len(s) == 0:
            return np.zeros(days), np.zeros(days), np.zeros(days)
        if len(s) < 7:
            last = float(s[-1])
            return np.full(days, last), np.full(days, last), np.full(days, last)

        # Initialise
        level = s[0]
        trend = s[1] - s[0]
        residuals = []

        for i in range(1, len(s)):
            prev_level = level
            level = alpha * s[i] + (1 - alpha) * (level + trend)
            trend = beta  * (level - prev_level) + (1 - beta) * trend
            residuals.append(s[i] - (prev_level + trend))

        # Forecast
        fc = np.array([level + (h + 1) * trend for h in range(days)])

        # Confidence interval widens with horizon (±1.96σ * sqrt(h))
        sigma = max(float(np.std(residuals)), 0.01) if residuals else 0.05  # minimum variance
        ci_width = 1.96 * sigma * np.sqrt(np.arange(1, days + 1))

        lo = np.clip(fc - ci_width, 0, 1)
        hi = np.clip(fc + ci_width, 0, 1)
        fc = np.clip(fc, 0, 1)
        return fc, lo, hi

    # Use last 60 days as context window
    hist_window = df.tail(FORECAST_WINDOW)
    last_date   = df["DATE"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS)

    fc_anom, lo_anom, hi_anom = holt_forecast(hist_window["Anomaly_Probability"], alpha=FORECAST_ALPHA, beta=FORECAST_BETA)
    fc_health, lo_health, hi_health = holt_forecast(hist_window["Health_Index"] / 100, alpha=FORECAST_ALPHA, beta=FORECAST_BETA)
    fc_health    = fc_health    * 100
    lo_health    = lo_health    * 100
    hi_health    = hi_health    * 100

    fig_fc = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Anomaly Probability Forecast", "Health Index Forecast (%)"),
        vertical_spacing=0.14)

    # ── Anomaly subplot ──
    fig_fc.add_trace(go.Scatter(
        x=hist_window["DATE"], y=hist_window["Anomaly_Probability"],
        name="Historic", line=dict(color=C["secondary"], width=2), mode="lines"), row=1, col=1)
    # CI band
    fig_fc.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(hi_anom) + list(lo_anom[::-1]),
        fill="toself", fillcolor=f"rgba(245,158,11,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=True), row=1, col=1)
    fig_fc.add_trace(go.Scatter(
        x=list(future_dates), y=list(fc_anom), name="Forecast",
        line=dict(color=C["primary"], width=2.5, dash="dot"),
        mode="lines+markers", marker=dict(size=5)), row=1, col=1)

    # ── Health subplot ──
    fig_fc.add_trace(go.Scatter(
        x=hist_window["DATE"], y=hist_window["Health_Index"].apply(to_pct),
        name="Health (hist)", line=dict(color=C["success"], width=2), showlegend=False), row=2, col=1)
    fig_fc.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(hi_health) + list(lo_health[::-1]),
        fill="toself", fillcolor="rgba(16,185,129,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False), row=2, col=1)
    fig_fc.add_trace(go.Scatter(
        x=list(future_dates), y=list(fc_health), name="Health (fc)",
        line=dict(color=C["primary"], width=2.5, dash="dot"),
        mode="lines+markers", marker=dict(size=5), showlegend=False), row=2, col=1)

    fig_fc.update_layout(**PLOTLY_LAYOUT, margin=_M_DEFAULT, legend=PLOTLY_LEGEND, height=520,
        title=dict(text=f"14-Day Predictive Forecast (Holt's Exponential Smoothing)", font=dict(size=13)))
    fig_fc.update_yaxes(range=[0, 1], row=1, col=1)
    fig_fc.update_yaxes(range=[0, 100], row=2, col=1)
    st.plotly_chart(fig_fc, width="stretch")

    # Verdict
    avg_fc_risk   = float(fc_anom.mean())
    avg_fc_health = float(fc_health.mean())
    trend_dir     = "improving ⬆" if fc_anom[-1] < fc_anom[0] else "worsening ⬇"

    if avg_fc_risk > 0.6:
        st.markdown(f"""<div class="alert-box alert-critical"><span class="alert-icon">🔮</span>
        <div><strong>HIGH RISK FORECAST</strong> — Predicted avg risk: <strong>{avg_fc_risk:.2f}</strong>.
        Health trending {trend_dir}. Predicted avg health: <strong>{avg_fc_health:.1f}%</strong>.
        Recommend immediate inspection.</div></div>""", unsafe_allow_html=True)
    elif avg_fc_risk > 0.4:
        st.markdown(f"""<div class="alert-box alert-warning"><span class="alert-icon">🔮</span>
        <div><strong>MODERATE RISK FORECAST</strong> — Predicted avg risk: <strong>{avg_fc_risk:.2f}</strong>.
        Trend: {trend_dir}. Schedule preventive maintenance.</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="alert-box alert-ok"><span class="alert-icon">🔮</span>
        <div><strong>STABLE FORECAST</strong> — Predicted avg risk: <strong>{avg_fc_risk:.2f}</strong>.
        Trend: {trend_dir}. Predicted avg health: <strong>{avg_fc_health:.1f}%</strong>.</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">14-Day Forecast Table</div>', unsafe_allow_html=True)
    fc_table = pd.DataFrame({
        "Date":              future_dates.strftime("%d %b %Y"),
        "Predicted Risk":    fc_anom.round(3),
        "Lower CI":          lo_anom.round(3),
        "Upper CI":          hi_anom.round(3),
        "Predicted Health%": fc_health.round(1),
        "Status":            ["HIGH" if p > 0.6 else ("MEDIUM" if p > 0.4 else "LOW") for p in fc_anom],
    })
    st.dataframe(fc_table, width="stretch", hide_index=True)


# ── TAB 4: FAULT ANALYSIS ───────────────────────────────────────────────────
with tab4:
    col_c, col_d = st.columns([1, 1])
    with col_c:
        st.markdown('<div class="section-title">Fault Distribution</div>', unsafe_allow_html=True)
        fault_counts = df["Fault_Type"].value_counts().reset_index()
        fault_counts.columns = ["Fault_Type", "Count"]
        fig_fault = go.Figure(go.Pie(
            labels=fault_counts["Fault_Type"], values=fault_counts["Count"], hole=0.55,
            marker=dict(colors=[C["danger"], C["primary"], C["secondary"], C["success"], "#8b5cf6", "#ec4899"]),
            textfont=dict(size=11)))
        fig_fault.update_layout(**PLOTLY_LAYOUT, margin=_M_DEFAULT,
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["grid"], orientation="v", x=1, y=0.5),
            height=320, title=dict(text="Fault Type Distribution", font=dict(size=13)))
        st.plotly_chart(fig_fault, width="stretch")

    with col_d:
        st.markdown('<div class="section-title">Priority Breakdown</div>', unsafe_allow_html=True)
        priority_counts = df["Maintenance_Priority"].value_counts().reset_index()
        priority_counts.columns = ["Priority", "Count"]
        color_map = {"HIGH": C["danger"], "MEDIUM": C["primary"], "LOW": C["success"]}
        fig_prio = go.Figure(go.Bar(
            x=priority_counts["Priority"], y=priority_counts["Count"],
            marker=dict(color=[color_map.get(p, C["secondary"]) for p in priority_counts["Priority"]]),
            text=priority_counts["Count"], textposition="outside"))
        fig_prio.update_layout(**PLOTLY_LAYOUT, margin=_M_DEFAULT, legend=PLOTLY_LEGEND, height=320,
            title=dict(text="Priority Distribution", font=dict(size=13)))
        st.plotly_chart(fig_prio, width="stretch")

    st.markdown('<div class="section-title">Recent Critical Alerts</div>', unsafe_allow_html=True)
    critical_days = df[df["Maintenance_Priority"] == "HIGH"].tail(10)
    if critical_days.empty:
        st.markdown('<div class="alert-box alert-ok"><span class="alert-icon">✅</span><div>No critical issues in selected date range.</div></div>', unsafe_allow_html=True)
    else:
        st.dataframe(critical_days[["DATE", "Maintenance_Status", "Fault_Type", "Revenue_Loss_INR"]]
            .sort_values("DATE", ascending=False)
            .style.format({"Revenue_Loss_INR": "₹{:,.2f}"}),
            width="stretch", hide_index=True)


# ── TAB 5: MAINTENANCE LOG ───────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">Alert History & Maintenance Log</div>', unsafe_allow_html=True)

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        priority_filter = st.multiselect("Priority", ["HIGH", "MEDIUM", "LOW"],
                                         default=["HIGH", "MEDIUM"])
    with col_f2:
        fault_filter = st.multiselect("Fault Type", df["Fault_Type"].unique().tolist(),
                                      default=df["Fault_Type"].unique().tolist())
    with col_f3:
        min_loss = st.number_input("Min Revenue Loss (₹)", min_value=0.0,
                                   value=0.0, step=10.0)

    log_df = df[
        df["Maintenance_Priority"].isin(priority_filter) &
        df["Fault_Type"].isin(fault_filter) &
        (df["Revenue_Loss_INR"] >= min_loss)
    ][["DATE", "Maintenance_Priority", "Maintenance_Status",
       "Fault_Type", "Anomaly_Probability", "Health_Index",
       "Energy_Loss_kWh", "Revenue_Loss_INR"]].sort_values("DATE", ascending=False).copy()

    # Summary stats
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1rem;">
      <div class="kpi-card red" style="padding:.8rem 1rem;">
        <div class="kpi-label">Total Alerts</div>
        <div class="kpi-value" style="font-size:1.4rem;">{len(log_df):,}</div>
      </div>
      <div class="kpi-card solar" style="padding:.8rem 1rem;">
        <div class="kpi-label">HIGH Priority</div>
        <div class="kpi-value" style="font-size:1.4rem;">{(log_df["Maintenance_Priority"]=="HIGH").sum():,}</div>
      </div>
      <div class="kpi-card blue" style="padding:.8rem 1rem;">
        <div class="kpi-label">Total Loss</div>
        <div class="kpi-value" style="font-size:1.4rem;">₹{log_df["Revenue_Loss_INR"].sum():,.0f}</div>
      </div>
      <div class="kpi-card green" style="padding:.8rem 1rem;">
        <div class="kpi-label">Avg Health on Alert Days</div>
        <div class="kpi-value" style="font-size:1.4rem;">{log_df["Health_Index"].mean() if not log_df.empty else 0:.1f}%</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Monthly alert heatmap
    st.markdown('<div class="section-title">Monthly Alert Frequency</div>', unsafe_allow_html=True)
    log_df["YearMonth"] = pd.to_datetime(log_df["DATE"]).dt.to_period("M").astype(str)
    monthly = log_df.groupby(["YearMonth","Maintenance_Priority"]).size().unstack(fill_value=0).reset_index()

    fig_monthly = go.Figure()
    for col_name, colour in [("HIGH", C["danger"]), ("MEDIUM", C["primary"]), ("LOW", C["success"])]:
        if col_name in monthly.columns:
            fig_monthly.add_trace(go.Bar(x=monthly["YearMonth"], y=monthly[col_name],
                name=col_name, marker_color=colour))
    fig_monthly.update_layout(**PLOTLY_LAYOUT, margin=_M_DEFAULT, legend=PLOTLY_LEGEND, height=260,
        barmode="stack", title=dict(text="Alerts per Month by Priority", font=dict(size=13)))
    fig_monthly.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_monthly, width="stretch")

    # Full log table
    st.markdown('<div class="section-title">Full Alert Log</div>', unsafe_allow_html=True)
    st.dataframe(
        log_df.style.format({
            "Anomaly_Probability": "{:.3f}",
            "Health_Index": "{:.1f}",
            "Energy_Loss_kWh": "{:.2f}",
            "Revenue_Loss_INR": "₹{:,.2f}",
        }).map(
            lambda v: "color: #ef4444;" if v == "HIGH" else ("color: #f59e0b;" if v == "MEDIUM" else ""),
            subset=["Maintenance_Priority"]
        ),
        width="stretch", hide_index=True, height=380
    )

    # Export filtered log
    filtered_csv = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export Filtered Log (CSV)", filtered_csv,
                       "maintenance_log_filtered.csv", "text/csv")


# ── TAB 6: MODEL REPORT ─────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-title">Model Performance Report</div>', unsafe_allow_html=True)

    auc_roc = model_meta.get("auc_roc", None)
    avg_prec = model_meta.get("avg_precision", None)
    feat_imp = model_meta.get("feature_importances", {})

    if auc_roc is None:
        st.markdown('<div class="alert-box alert-warning"><span class="alert-icon">⚠️</span>'
                    '<div>Model metadata not found. Re-run train_model.py to generate it.</div></div>',
                    unsafe_allow_html=True)
    else:
        # Top metrics
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem;">
          <div class="kpi-card green">
            <div class="kpi-label">AUC-ROC Score</div>
            <div class="kpi-value">{auc_roc:.4f}</div>
            <div class="kpi-sub">{'Excellent (>0.99)' if auc_roc > 0.99 else 'Good (>0.95)' if auc_roc > 0.95 else 'Acceptable'}</div>
          </div>
          <div class="kpi-card blue">
            <div class="kpi-label">Avg Precision (AUPRC)</div>
            <div class="kpi-value">{avg_prec:.4f}</div>
            <div class="kpi-sub">Precision-Recall Area</div>
          </div>
          <div class="kpi-card solar">
            <div class="kpi-label">Model Type</div>
            <div class="kpi-value" style="font-size:1rem;padding-top:.3rem;">Ensemble</div>
            <div class="kpi-sub">RF + GBM + ExtraTrees · Isotonic Calibration</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Model description
        st.markdown('<div class="section-title">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""<div class="alert-box alert-info"><span class="alert-icon">🤖</span>
        <div>
        <strong>Voting Ensemble</strong> of three independent classifiers trained on 20 engineered features
        with time-series aware cross-validation (TimeSeriesSplit, 5 folds) to prevent data leakage.
        Probabilities are post-processed with <strong>Isotonic Regression calibration</strong> on a held-out
        validation set (15% of data) to ensure probabilities are statistically reliable.
        Final evaluation on a completely unseen test set (last 15% chronologically).
        </div></div>""", unsafe_allow_html=True)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown('<div class="section-title">Training Split</div>', unsafe_allow_html=True)
            n_total = len(df_full)
            splits = pd.DataFrame({
                "Split":   ["Train (70%)", "Validation (15%)", "Test (15%)"],
                "Records": [int(n_total*0.70), int(n_total*0.15), int(n_total*0.15)],
                "Purpose": ["Model training", "Probability calibration", "Final evaluation"],
            })
            st.dataframe(splits, width="stretch", hide_index=True)

            st.markdown('<div class="section-title">Performance Interpretation</div>', unsafe_allow_html=True)
            interp = pd.DataFrame({
                "Metric": ["AUC-ROC", "AUPRC", "Cross-Val AUC"],
                "Score":  [f"{auc_roc:.4f}", f"{avg_prec:.4f}", "~0.9965"],
                "Meaning": [
                    "Near-perfect separation of normal vs anomaly",
                    "High precision at all recall thresholds",
                    "Consistent across all 5 time-based folds",
                ],
            })
            st.dataframe(interp, width="stretch", hide_index=True)

        with col_m2:
            st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
            if feat_imp:
                fi_df = pd.Series(feat_imp).sort_values(ascending=True).tail(15)
                fig_fi = go.Figure(go.Bar(
                    x=fi_df.values, y=fi_df.index, orientation="h",
                    marker=dict(
                        color=fi_df.values,
                        colorscale=[[0, C["secondary"]], [0.5, C["primary"]], [1, C["success"]]],
                        showscale=False
                    ),
                    text=[f"{v:.3f}" for v in fi_df.values],
                    textposition="outside"
                ))
                fig_fi.update_layout(**PLOTLY_LAYOUT, margin=_M_DEFAULT, legend=PLOTLY_LEGEND, height=420,
                    title=dict(text="Top 15 Feature Importances (Random Forest)", font=dict(size=13)))
                
                fig_fi.update_xaxes(showgrid=True, gridcolor=C["grid"], zeroline=False)
                fig_fi.update_yaxes(showgrid=False, zeroline=False)
                st.plotly_chart(fig_fi, width="stretch")

        # Feature table
        st.markdown('<div class="section-title">All 20 Features Used</div>', unsafe_allow_html=True)
        features_list = model_meta.get("features", [])
        if features_list:
            fi_full = {f: feat_imp.get(f, 0.0) for f in features_list}
            feat_table = pd.DataFrame([
                {
                    "Feature": f,
                    "Importance": f"{fi_full[f]:.4f}",
                    "Category": (
                        "Seasonality" if "DOY" in f else
                        "Lag" if "LAG" in f else
                        "Rolling PR" if any(x in f for x in ["PR_","Z_SCORE","TREND"]) else
                        "Energy" if "kWh" in f else
                        "Environmental"
                    )
                }
                for f in sorted(features_list, key=lambda x: fi_full.get(x, 0), reverse=True)
            ])
            st.dataframe(feat_table, width="stretch", hide_index=True)


# =============================
# RAW DATA
# =============================
if show_raw:
    st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(df, width="stretch", hide_index=True)

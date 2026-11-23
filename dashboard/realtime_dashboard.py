"""
dashboard/realtime_dashboard.py
==================================
Live real-time model monitoring dashboard built with Streamlit.

Displays rolling error rate, drift status, fairness gap,
and alert history — auto-refreshes every 30 seconds.

Run:
    streamlit run dashboard/realtime_dashboard.py
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "MLens Monitor — Real-Time Dashboard",
    page_icon   = "📡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #1a1a2e; }
  [data-testid="stSidebar"]          { background-color: #16213e; }
  .health-ok   { color: #639922; font-weight: 600; font-size: 20px; }
  .health-warn { color: #EF9F27; font-weight: 600; font-size: 20px; }
  .health-crit { color: #E24B4A; font-weight: 600; font-size: 20px; }
  .metric-card {
    background: #16213e; border-radius: 10px;
    padding: 1rem 1.2rem; border: 1px solid #333355;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📡 MLens Monitor")
    st.markdown("---")

    st.subheader("⚙️ Simulation Settings")
    model_name    = st.text_input("Model name", value="MyProductionModel")
    n_batches     = st.slider("Batches to simulate", 5, 50, 20)
    batch_size    = st.slider("Predictions per batch", 10, 200, 50)
    error_rate    = st.slider("Base error rate", 0.0, 0.5, 0.08)
    drift_at      = st.slider("Introduce drift at batch #", 1, n_batches, n_batches // 2)
    drift_error   = st.slider("Error rate after drift", 0.0, 0.8, 0.35)
    window_size   = st.slider("Monitor window size", 100, 2000, 500)
    auto_refresh  = st.checkbox("Auto-refresh (30s)", value=False)

    st.markdown("---")
    run_btn = st.button("▶️ Run Simulation", use_container_width=True, type="primary")
    reset_btn = st.button("🔄 Reset", use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────

st.title("📡 MLens Real-Time Model Monitor")
st.caption(f"Model: **{model_name}** · Live rolling metrics · github.com/saiganesh47/mlens")

if reset_btn:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if not run_btn and "snapshots" not in st.session_state:
    st.info("👈 Configure settings in the sidebar and click **Run Simulation**.")
    st.stop()

# ── Run or load simulation ─────────────────────────────────────────────────

if run_btn or "snapshots" not in st.session_state:
    from mlens.monitoring.realtime_monitor import RealtimeMonitor

    monitor   = RealtimeMonitor(
        model_name           = model_name,
        window_size          = window_size,
        error_rate_threshold = 0.20,
        dp_gap_threshold     = 0.10,
    )
    snapshots = []
    rng       = np.random.default_rng(42)

    with st.spinner("Simulating prediction stream …"):
        for i in range(n_batches):
            # Introduce drift after drift_at batches
            current_err = drift_error if i >= drift_at else error_rate
            y_true = rng.integers(0, 2, batch_size)
            y_pred = y_true.copy()
            flip   = rng.choice(batch_size,
                                size=int(current_err * batch_size),
                                replace=False)
            y_pred[flip] = 1 - y_pred[flip]
            sensitive = rng.choice(["Group A", "Group B"], size=batch_size)
            # Bias group B after drift
            if i >= drift_at:
                b_mask = sensitive == "Group B"
                b_flip = np.where(b_mask)[0][:int(len(b_mask.sum()) * 0.3 + 1)]
                y_pred[b_flip] = 1

            snap = monitor.ingest(y_true, y_pred, sensitive=sensitive)
            snapshots.append({
                "batch":          i + 1,
                "timestamp":      snap.timestamp,
                "error_rate":     snap.error_rate,
                "drift_detected": snap.drift_detected,
                "dp_gap":         snap.dp_gap or 0.0,
                "n_predictions":  snap.n_predictions,
                "alert_count":    snap.alert_count,
            })

    st.session_state["snapshots"] = snapshots
    st.session_state["monitor"]   = monitor

snapshots = st.session_state["snapshots"]
monitor   = st.session_state["monitor"]
df        = pd.DataFrame(snapshots)
latest    = monitor.status()


# ── KPI row ────────────────────────────────────────────────────────────────

st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)

health = ("🔴 Critical" if latest.drift_detected or latest.error_rate > 0.30
          else "🟡 Warning"  if latest.error_rate > 0.20
          else "🟢 Healthy")
health_class = ("health-crit" if "Critical" in health
                else "health-warn" if "Warning" in health
                else "health-ok")

c1.metric("Health",       health)
c2.metric("Error Rate",   f"{latest.error_rate:.2%}")
c3.metric("Total Preds",  f"{latest.n_predictions:,}")
c4.metric("Drift",        "⚠️ Detected" if latest.drift_detected else "✅ Stable")
c5.metric("Alerts Fired", latest.alert_count)

st.markdown("---")

# ── Error rate chart ───────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Rolling Error Rate")
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=df["batch"], y=df["error_rate"],
        mode="lines+markers",
        line=dict(color="#7F77DD", width=2),
        marker=dict(
            color=["#E24B4A" if d else "#7F77DD" for d in df["drift_detected"]],
            size=8,
        ),
        name="Error Rate",
        hovertemplate="Batch %{x}<br>Error: %{y:.3f}<extra></extra>",
    ))
    fig_err.add_hline(
        y=0.20, line_dash="dash", line_color="#EF9F27",
        annotation_text="Warning (0.20)",
        annotation_font_color="#EF9F27",
    )
    fig_err.add_hline(
        y=0.30, line_dash="dash", line_color="#E24B4A",
        annotation_text="Critical (0.30)",
        annotation_font_color="#E24B4A",
    )
    fig_err.update_layout(
        paper_bgcolor="#16213e", plot_bgcolor="#16213e",
        font=dict(color="#e8e8e8"),
        xaxis=dict(title="Batch #", gridcolor="#333355"),
        yaxis=dict(title="Error Rate", gridcolor="#333355", tickformat=".0%"),
        height=320, margin=dict(l=20, r=20, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_err, use_container_width=True)

with col2:
    st.subheader("⚖️ Rolling Demographic Parity Gap")
    fig_dp = go.Figure()
    fig_dp.add_trace(go.Scatter(
        x=df["batch"], y=df["dp_gap"],
        mode="lines+markers",
        line=dict(color="#1D9E75", width=2),
        marker=dict(size=6, color="#1D9E75"),
        hovertemplate="Batch %{x}<br>DP Gap: %{y:.3f}<extra></extra>",
    ))
    fig_dp.add_hline(
        y=0.10, line_dash="dash", line_color="#EF9F27",
        annotation_text="Threshold (0.10)",
        annotation_font_color="#EF9F27",
    )
    fig_dp.update_layout(
        paper_bgcolor="#16213e", plot_bgcolor="#16213e",
        font=dict(color="#e8e8e8"),
        xaxis=dict(title="Batch #", gridcolor="#333355"),
        yaxis=dict(title="DP Gap", gridcolor="#333355"),
        height=320, margin=dict(l=20, r=20, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_dp, use_container_width=True)


# ── Drift timeline ─────────────────────────────────────────────────────────

st.subheader("📊 Drift Detection Timeline")
drift_batches = df[df["drift_detected"]]["batch"].tolist()
if drift_batches:
    st.warning(f"⚠️ Concept drift detected at batches: {drift_batches}")
else:
    st.success("✅ No concept drift detected in this simulation run.")

fig_cum = go.Figure()
fig_cum.add_trace(go.Bar(
    x=df["batch"], y=df["n_predictions"],
    marker=dict(
        color=["#E24B4A" if d else "#534AB7" for d in df["drift_detected"]],
        line=dict(width=0),
    ),
    hovertemplate="Batch %{x}<br>Total preds: %{y:,}<extra></extra>",
))
fig_cum.update_layout(
    paper_bgcolor="#16213e", plot_bgcolor="#16213e",
    font=dict(color="#e8e8e8"),
    xaxis=dict(title="Batch #", gridcolor="#333355"),
    yaxis=dict(title="Cumulative predictions", gridcolor="#333355"),
    height=220, margin=dict(l=20, r=20, t=10, b=40),
)
st.plotly_chart(fig_cum, use_container_width=True)


# ── Raw snapshot table ─────────────────────────────────────────────────────

with st.expander("📋 Raw Snapshot Table"):
    display_df = df.copy()
    display_df["error_rate"] = display_df["error_rate"].map("{:.3f}".format)
    display_df["dp_gap"]     = display_df["dp_gap"].map("{:.3f}".format)
    st.dataframe(display_df, use_container_width=True)


# ── Auto-refresh ───────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(30)
    st.rerun()

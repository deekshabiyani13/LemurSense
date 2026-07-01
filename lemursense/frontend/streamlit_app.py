"""
streamlit_app.py
------------------
LemurSense dashboard. Talks to the FastAPI backend to:
  - advance the simulated monitoring clock
  - visualize per-host metrics with anomaly markers
  - show the "Lemur Troop" attention panel (who's the sentinel, how
    fast is each metric being polled, troop-wide alert level)
  - show SHAP explanations for the latest anomaly on a selected host
"""

import os
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = os.environ.get("LEMURSENSE_API_URL", "http://localhost:8000")

st.set_page_config(page_title="LemurSense", page_icon="🐒", layout="wide")


# ---------------------------------------------------------------- helpers
def api_get(path, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path, **params):
    r = requests.post(f"{API_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def alert_color(level):
    if level < 0.3:
        return "#4caf50"
    if level < 0.6:
        return "#ff9800"
    return "#e53935"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🐒 LemurSense")
st.sidebar.caption("Biomimetic adaptive monitoring, modeled on lemur vigilance behavior.")

try:
    hosts_info = api_get("/hosts")
    HOSTS = hosts_info["hosts"]
    METRICS = hosts_info["metrics"]
    backend_ok = True
except Exception as e:
    st.sidebar.error(f"Cannot reach backend at {API_URL}\n\n{e}")
    backend_ok = False
    HOSTS, METRICS = [], []

if backend_ok:
    selected_host = st.sidebar.selectbox("Host", HOSTS)

    st.sidebar.markdown("---")
    steps = st.sidebar.slider("Ticks to advance", 1, 50, 5)
    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("▶ Advance Time", use_container_width=True):
        api_post("/simulate/tick", steps=steps)
    if col_b.button("⟲ Reset", use_container_width=True):
        api_post("/reset")

    auto = st.sidebar.checkbox("Auto-advance (1 tick / refresh)")
    if auto:
        api_post("/simulate/tick", steps=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Biomimicry mapping**\n\n"
        "- 🦧 *Sentinel* = metric getting the most attention right now\n"
        "- 📣 *Alarm call* = anomaly on one metric raises attention on its siblings\n"
        "- 🍃 *Foraging* = polling interval shrinks/grows with attention\n"
        "- ☀️ *Circadian* = attention baseline shifts with time of day"
    )


# ---------------------------------------------------------------- main
if not backend_ok:
    st.title("LemurSense")
    st.warning("Start the backend with `uvicorn app:app --reload` from the backend/ folder, "
               "then set LEMURSENSE_API_URL if it's not on localhost:8000.")
    st.stop()

st.title("🐒 LemurSense — Biomimetic Adaptive Monitoring")

lemur_state = api_get("/lemur/state")
troop_level = lemur_state["troop_alert_level"]

top_cols = st.columns([1, 1, 1, 2])
top_cols[0].metric("Troop Alert Level", f"{troop_level:.2f}")
top_cols[1].metric("Hosts Monitored", len(HOSTS))
events = api_get("/events", limit=200)
flagged_count = sum(len(e["flagged"]) for e in events)
top_cols[2].metric("Anomalies Flagged (session)", flagged_count)

with top_cols[3]:
    st.markdown("**Troop status**")
    color = alert_color(troop_level)
    st.markdown(
        f"<div style='background:{color};color:white;padding:8px 14px;border-radius:8px;"
        f"display:inline-block;font-weight:600;'>"
        f"{'🟢 Calm troop' if troop_level < 0.3 else '🟠 Elevated vigilance' if troop_level < 0.6 else '🔴 Alarm state'}"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

left, right = st.columns([2, 1])

# ---- metrics chart
with left:
    st.subheader(f"📈 Metrics — {selected_host}")
    records = api_get(f"/metrics/{selected_host}", window=250)
    if records:
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = make_subplots(rows=2, cols=2, subplot_titles=METRICS, shared_xaxes=True)
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for metric, (r, c) in zip(METRICS, positions):
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df[metric], mode="lines", name=metric,
                           line=dict(width=1.5)),
                row=r, col=c,
            )
            anomalies = df[df["is_anomaly"]]
            if not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies["timestamp"], y=anomalies[metric], mode="markers",
                        name=f"{metric} anomaly", marker=dict(color="red", size=7, symbol="x"),
                        showlegend=False,
                    ),
                    row=r, col=c,
                )
        fig.update_layout(height=550, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data revealed yet — click 'Advance Time' in the sidebar.")

    st.subheader("🚨 Recent flagged events")
    flat_events = []
    for e in events[-30:]:
        for f in e["flagged"]:
            flat_events.append(f)
    if flat_events:
        st.dataframe(pd.DataFrame(flat_events).sort_values("timestamp", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No anomalies flagged yet this session.")

# ---- lemur troop panel
with right:
    st.subheader("🦧 Lemur Troop Attention")
    sent_df = pd.DataFrame(lemur_state["sentinels"])
    sent_df = sent_df[sent_df["host"] == selected_host].sort_values("attention", ascending=False)

    for _, row in sent_df.iterrows():
        icon = "👁️" if row["is_sentinel"] else "😴"
        st.markdown(f"{icon} **{row['metric']}**")
        st.progress(min(1.0, max(0.0, row["attention"])))
        st.caption(
            f"attention={row['attention']:.2f} · "
            f"polling every {row['polling_interval_sec']:.0f}s · "
            f"last score={row['last_score']:.2f}"
        )

    st.markdown("---")
    st.subheader("🔍 Explain latest point (SHAP)")
    if st.button("Explain most recent reading", use_container_width=True):
        with st.spinner("Consulting the sentinel... (computing SHAP values)"):
            try:
                explanation = api_get(f"/explain/{selected_host}")
                st.caption(
                    f"timestamp: {explanation['timestamp']} · "
                    f"true label: {'anomaly' if explanation['true_label'] else 'normal'} "
                    f"({explanation['true_type'] or 'n/a'})"
                )
                contrib_df = pd.DataFrame(explanation["contributions"])
                fig2 = go.Figure(go.Bar(
                    x=contrib_df["shap_value"],
                    y=contrib_df["feature"],
                    orientation="h",
                    marker=dict(
                        color=["#e53935" if v > 0 else "#4caf50" for v in contrib_df["shap_value"]]
                    ),
                ))
                fig2.update_layout(
                    height=350, margin=dict(t=20, b=20),
                    xaxis_title="SHAP value (push toward anomaly →)",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Could not compute explanation: {e}")

st.markdown("---")
st.caption(
    "LemurSense — a biomimetic explainable AI system for IT anomaly monitoring. "
    "Backend: FastAPI + Isolation Forest + SHAP. Frontend: Streamlit."
)

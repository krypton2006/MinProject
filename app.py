import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AI Heartbeat System", layout="wide")

# ---------------- HEADER ----------------
st.markdown("## 💓 AI Heartbeat Monitoring System")
st.markdown("Real-time + Historical + AI Analysis")

# ---------------- CHECK FILE ----------------
if not os.path.exists("heartbeat.csv"):
    st.error("❌ heartbeat.csv not found. Run generator.py first.")
    st.stop()

df = pd.read_csv("heartbeat.csv", names=["time", "hb"])

if df.empty:
    st.warning("⚠️ No data yet...")
    st.stop()

df["hb"] = df["hb"].astype(int)

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("⚙️ Controls")

window = st.sidebar.slider("Moving Average Window", 3, 20, 10)
anomaly_rate = st.sidebar.slider("Anomaly Sensitivity", 0.01, 0.2, 0.05)

# ---------------- CURRENT METRICS ----------------
latest = df["hb"].iloc[-1]
avg = df["hb"].mean()
max_hb = df["hb"].max()
min_hb = df["hb"].min()

col1, col2, col3, col4 = st.columns(4)

col1.metric("💓 Current BPM", latest)
col2.metric("📊 Average", round(avg, 2))
col3.metric("🔺 Max", max_hb)
col4.metric("🔻 Min", min_hb)

# ---------------- STATUS ----------------
def classify(hb):
    if hb < 60:
        return "Low"
    elif hb <= 100:
        return "Normal"
    else:
        return "High"

status = classify(latest)

if status == "High":
    st.error("🚨 High Heart Rate Detected")
elif status == "Low":
    st.warning("⚠️ Low Heart Rate Detected")
else:
    st.success("✅ Normal Heart Rate")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Trends",
    "🧠 AI Analysis",
    "📊 Statistics",
    "📋 Data Explorer"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("Live Heartbeat Signal")

    st.line_chart(df["hb"])

    df["avg"] = df["hb"].rolling(window).mean()

    st.subheader("Smoothed Signal (Moving Avg)")
    st.line_chart(df[["hb", "avg"]])

# ================= TAB 2 =================
with tab2:
    st.subheader("Anomaly Detection (AI)")

    model = IsolationForest(contamination=anomaly_rate)
    df["anomaly"] = model.fit_predict(df[["hb"]])

    anomalies = df[df["anomaly"] == -1]

    st.write(f"Detected {len(anomalies)} anomalies")

    st.dataframe(anomalies.tail(20))

    # Highlight anomalies in chart
    chart_data = df.copy()
    chart_data["anomaly_flag"] = chart_data["anomaly"].apply(lambda x: 1 if x == -1 else 0)

    st.line_chart(chart_data[["hb", "anomaly_flag"]])

# ================= TAB 3 =================
with tab3:
    st.subheader("Distribution Analysis")

    hist = np.histogram(df["hb"], bins=15)[0]
    st.bar_chart(hist)

    st.subheader("Statistical Summary")
    st.write(df["hb"].describe())

    st.subheader("Trend Direction")

    if len(df) > 10:
        if df["hb"].iloc[-1] > df["hb"].iloc[-10]:
            st.success("📈 Increasing Trend")
        else:
            st.info("📉 Decreasing / Stable Trend")

# ================= TAB 4 =================
with tab4:
    st.subheader("Raw Data")

    st.dataframe(df.tail(100))

    st.download_button(
        "⬇️ Download Data",
        df.to_csv(index=False),
        file_name="heartbeat_data.csv"
    )

# ---------------- AUTO REFRESH ----------------

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# ---------------- AUTO REFRESH ----------------
st_autorefresh(interval=2000, key="refresh")

st.set_page_config(page_title="AI Heartbeat System", layout="wide")

# ---------------- HEADER ----------------
st.markdown("## 💓 AI Heartbeat Monitoring System")
st.markdown("Real-time + AI + Sleep + Analytics + Prediction")

# ---------------- LIVE INDICATOR ----------------
if int(time.time()) % 2 == 0:
    st.markdown("<h4 style='color:red;'>● LIVE</h4>", unsafe_allow_html=True)
else:
    st.markdown("<h4 style='color:gray;'>● LIVE</h4>", unsafe_allow_html=True)

# ---------------- CHECK FILE ----------------
if not os.path.exists("heartbeat.csv"):
    st.error("heartbeat.csv not found. Run generator.py first.")
    st.stop()

df = pd.read_csv("heartbeat.csv", names=["time", "hb"])

if df.empty:
    st.warning("No data yet...")
    st.stop()

# ---------------- PREPROCESS ----------------
df["hb"] = df["hb"].astype(int)
df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
window = st.sidebar.slider("Moving Avg Window", 3, 20, 10)
anomaly_rate = st.sidebar.slider("Anomaly Sensitivity", 0.01, 0.2, 0.05)

# ---------------- METRICS ----------------
latest = df["hb"].iloc[-1]
avg = df["hb"].mean()
max_hb = df["hb"].max()
min_hb = df["hb"].min()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current BPM", latest)
col2.metric("Average", round(avg, 2))
col3.metric("Max", max_hb)
col4.metric("Min", min_hb)

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
    st.error("High Heart Rate Detected")
elif status == "Low":
    st.warning("Low Heart Rate Detected")
else:
    st.success("Normal Heart Rate")

# =====================================================
# 💤 SLEEP DETECTION
# =====================================================
df["rolling_avg"] = df["hb"].rolling(30).mean()
df["rolling_std"] = df["hb"].rolling(30).std()

def detect_sleep(row):
    if row["rolling_avg"] < 65 and row["rolling_std"] < 3:
        return "Sleep"
    else:
        return "Awake"

df["state"] = df.apply(detect_sleep, axis=1)

current_state = df["state"].iloc[-1]

if current_state == "Sleep":
    st.info("User is likely sleeping")
else:
    st.success("User is awake")

sleep_time = (df["state"] == "Sleep").sum()
st.metric("Sleep Duration (sec)", sleep_time)

if df["rolling_std"].mean() < 4:
    st.success("Good Sleep Quality")
else:
    st.warning("Disturbed Sleep")

# =====================================================
# 📅 DATE ANALYSIS
# =====================================================
daily_stats = df.groupby("date").agg({
    "hb": ["mean", "max", "min"],
    "state": lambda x: (x == "Sleep").sum()
})

daily_stats.columns = ["avg_hr", "max_hr", "min_hr", "sleep_duration"]
daily_stats = daily_stats.reset_index()

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Live Trends",
    "AI Analysis",
    "Statistics",
    "Data Explorer",
    "Sleep Analysis",
    "Date Analysis",
    "Prediction"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("Live Heartbeat")
    st.line_chart(df["hb"])

    df["avg"] = df["hb"].rolling(window).mean()
    st.subheader("Smoothed Signal")
    st.line_chart(df[["hb", "avg"]])

# ================= TAB 2 =================
with tab2:
    st.subheader("Anomaly Detection")

    model = IsolationForest(contamination=anomaly_rate)
    df["anomaly"] = model.fit_predict(df[["hb"]])

    anomalies = df[df["anomaly"] == -1]
    st.write("Detected anomalies:", len(anomalies))
    st.dataframe(anomalies.tail(20))

    chart_data = df.copy()
    chart_data["flag"] = chart_data["anomaly"].apply(lambda x: 1 if x == -1 else 0)
    st.line_chart(chart_data[["hb", "flag"]])

# ================= TAB 3 =================
with tab3:
    st.subheader("Statistics")

    hist = np.histogram(df["hb"], bins=15)[0]
    st.bar_chart(hist)

    st.write(df["hb"].describe())

# ================= TAB 4 =================
with tab4:
    st.subheader("Raw Data")
    st.dataframe(df.tail(100))

    st.download_button(
        "Download Data",
        df.to_csv(index=False),
        file_name="heartbeat_data.csv"
    )

# ================= TAB 5 =================
with tab5:
    st.subheader("Sleep Pattern")
    st.line_chart(df[["hb", "rolling_avg"]])
    st.dataframe(df[["time", "hb", "state"]].tail(50))

# ================= TAB 6 =================
with tab6:
    st.subheader("Date Analysis")

    selected_date = st.selectbox("Select Date", daily_stats["date"])
    day_data = daily_stats[daily_stats["date"] == selected_date]

    if not day_data.empty:
        st.metric("Avg HR", round(day_data["avg_hr"].values[0], 2))
        st.metric("Max HR", day_data["max_hr"].values[0])
        st.metric("Min HR", day_data["min_hr"].values[0])
        st.metric("Sleep Duration", day_data["sleep_duration"].values[0])

    st.line_chart(daily_stats.set_index("date")["avg_hr"])

# ================= TAB 7 =================
with tab7:
    st.subheader("Prediction")

    if len(df) > 20:
        df["t"] = range(len(df))

        X = df[["t"]]
        y = df["hb"]

        model = LinearRegression()
        model.fit(X, y)

        future_t = np.array(range(len(df), len(df) + 20)).reshape(-1, 1)
        future_pred = model.predict(future_t)

        future_df = pd.DataFrame({
            "t": future_t.flatten(),
            "predicted_hb": future_pred
        })

        st.line_chart(future_df.set_index("t"))

        if future_pred.mean() > 100:
            st.error("Future Risk: High HR Expected")
        elif future_pred.mean() < 60:
            st.warning("Future Risk: Low HR Expected")
        else:
            st.success("Stable HR Expected")
    else:
        st.warning("Not enough data")

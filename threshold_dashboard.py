# ============================================================
# INTERACTIVE GOVERNANCE THRESHOLD DASHBOARD
# Fraud Detection Threshold Simulator
# ============================================================
# Run with:
# streamlit run threshold_dashboard.py
#
# Required packages:
# pip install streamlit pandas numpy scikit-learn matplotlib
# ============================================================

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Governance Threshold Simulator",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Governance Threshold Simulator for Fraud Detection")
st.markdown(
    """
This dashboard shows how changing the **decision threshold** affects fraud detection governance metrics:

- Alert volume
- Recall
- Precision
- F1-score
- AUC-PR
- ROC-AUC
- Approximate explanation latency

The model stays the same. Only the threshold changes.
"""
)

# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("Simulation Settings")

uploaded_file = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])

max_rows = st.sidebar.slider(
    "Maximum rows to use",
    min_value=5_000,
    max_value=100_000,
    value=30_000,
    step=5_000,
)

test_size = st.sidebar.slider(
    "Test size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
)

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.0000,
    max_value=1.0000,
    value=0.0500,
    step=0.0005,
    format="%.4f",
)

latency_per_explanation_ms = st.sidebar.slider(
    "Assumed explanation latency per alert (ms)",
    min_value=1,
    max_value=2000,
    value=250,
    step=10,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Governance Modes")
recall_target = st.sidebar.slider(
    "Recall target",
    min_value=0.50,
    max_value=1.00,
    value=0.90,
    step=0.01,
)

alert_rate_target = st.sidebar.slider(
    "Alert-rate target",
    min_value=0.001,
    max_value=0.100,
    value=0.010,
    step=0.001,
    format="%.3f",
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=True)
def load_data(file, max_rows):
    df = pd.read_csv(file)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a target column named 'Class'.")

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    return df


def add_sids_proxy_features(df):
    df = df.copy()

    if "Time" in df.columns:
        df["hour_of_day"] = ((df["Time"] // 3600) % 24).astype(int)
        df["is_night"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)
        df["is_business_hours"] = df["hour_of_day"].between(8, 17).astype(int)

    if "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])
        df["high_amount_flag"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)

    if "is_night" in df.columns and "high_amount_flag" in df.columns:
        df["night_high_amount"] = df["is_night"] * df["high_amount_flag"]

    return df

# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource(show_spinner=True)
def train_model(df, test_size):
    df = add_sids_proxy_features(df)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    train_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - train_start

    score_start = time.time()
    y_scores = model.predict_proba(X_test)[:, 1]
    scoring_time = time.time() - score_start

    return model, X_test, y_test, y_scores, training_time, scoring_time

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def compute_metrics(y_true, y_scores, threshold, latency_per_explanation_ms):
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    alert_volume = int(y_pred.sum())
    alert_rate = float(y_pred.mean())

    total_explanation_latency_sec = (alert_volume * latency_per_explanation_ms) / 1000

    return {
        "threshold": float(threshold),
        "auc_pr": float(average_precision_score(y_true, y_scores)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alert_volume": alert_volume,
        "alert_rate": alert_rate,
        "total_explanation_latency_sec": float(total_explanation_latency_sec),
        "latency_per_explanation_ms": int(latency_per_explanation_ms),
    }


def threshold_for_recall(y_true, y_scores, recall_target):
    thresholds = np.unique(np.quantile(y_scores, np.linspace(0, 1, 300)))
    candidates = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= recall_target:
            candidates.append((t, y_pred.sum()))

    if not candidates:
        return 0.5

    return float(sorted(candidates, key=lambda x: x[1])[0][0])


def threshold_for_alert_rate(y_scores, alert_rate_target):
    return float(np.quantile(y_scores, 1 - alert_rate_target))


def make_tradeoff_df(y_true, y_scores, latency_per_explanation_ms):
    thresholds = np.unique(np.quantile(y_scores, np.linspace(0, 1, 120)))
    rows = []

    for t in thresholds:
        rows.append(compute_metrics(y_true, y_scores, t, latency_per_explanation_ms))

    return pd.DataFrame(rows)

# ============================================================
# MAIN APP
# ============================================================

if uploaded_file is None:
    st.info("Upload the Kaggle creditcard.csv file from the sidebar to begin.")
    st.stop()

try:
    df = load_data(uploaded_file, max_rows)
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# Dataset overview
fraud_count = int(df["Class"].sum())
total_count = int(len(df))
fraud_rate = float(df["Class"].mean())

st.subheader("1. Dataset Governance Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{total_count:,}")
c2.metric("Fraud cases", f"{fraud_count:,}")
c3.metric("Fraud rate", f"{fraud_rate:.4%}")
c4.metric("Missing values", f"{int(df.isnull().sum().sum()):,}")

with st.spinner("Training lightweight model and scoring transactions..."):
    model, X_test, y_test, y_scores, training_time, scoring_time = train_model(df, test_size)

st.subheader("2. Model Status")

m1, m2, m3 = st.columns(3)
m1.metric("Training time", f"{training_time:.3f} sec")
m2.metric("Scoring time", f"{scoring_time:.4f} sec")
m3.metric("Test transactions", f"{len(y_test):,}")

# Current threshold metrics
metrics = compute_metrics(y_test, y_scores, threshold, latency_per_explanation_ms)

st.subheader("3. Live Threshold Impact")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Recall", f"{metrics['recall']:.3f}")
k2.metric("Precision", f"{metrics['precision']:.3f}")
k3.metric("Alert Volume", f"{metrics['alert_volume']:,}")
k4.metric("Alert Rate", f"{metrics['alert_rate']:.2%}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
k6.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
k7.metric("F1-score", f"{metrics['f1']:.3f}")
k8.metric("Total explanation latency", f"{metrics['total_explanation_latency_sec']:.2f} sec")

st.markdown("### Confusion Matrix at Selected Threshold")
cm_df = pd.DataFrame(
    {
        "Predicted Legitimate": [metrics["true_negatives"], metrics["false_negatives"]],
        "Predicted Fraud": [metrics["false_positives"], metrics["true_positives"]],
    },
    index=["Actual Legitimate", "Actual Fraud"],
)
st.dataframe(cm_df, use_container_width=True)

# Governance modes
st.subheader("4. Governance Operating Modes")

recall_threshold = threshold_for_recall(y_test, y_scores, recall_target)
alert_threshold = threshold_for_alert_rate(y_scores, alert_rate_target)

recall_metrics = compute_metrics(y_test, y_scores, recall_threshold, latency_per_explanation_ms)
alert_metrics = compute_metrics(y_test, y_scores, alert_threshold, latency_per_explanation_ms)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Recall-Based Mode")
    st.caption("Risk-control mode: catch at least the target proportion of fraud.")
    st.metric("Selected threshold", f"{recall_threshold:.4f}")
    st.metric("Recall", f"{recall_metrics['recall']:.3f}")
    st.metric("Alert volume", f"{recall_metrics['alert_volume']:,}")
    st.metric("False negatives", f"{recall_metrics['false_negatives']:,}")

with col_b:
    st.markdown("### Alert-Volume Mode")
    st.caption("Operational-control mode: limit alerts to analyst capacity.")
    st.metric("Selected threshold", f"{alert_threshold:.4f}")
    st.metric("Recall", f"{alert_metrics['recall']:.3f}")
    st.metric("Alert volume", f"{alert_metrics['alert_volume']:,}")
    st.metric("False negatives", f"{alert_metrics['false_negatives']:,}")

# Trade-off curves
st.subheader("5. Trade-Off Curves")

tradeoff_df = make_tradeoff_df(y_test, y_scores, latency_per_explanation_ms)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(tradeoff_df["alert_volume"], tradeoff_df["recall"])
ax1.scatter([recall_metrics["alert_volume"]], [recall_metrics["recall"]], label="Recall mode")
ax1.scatter([alert_metrics["alert_volume"]], [alert_metrics["recall"]], label="Alert-volume mode")
ax1.set_xlabel("Alert Volume")
ax1.set_ylabel("Recall")
ax1.set_title("Recall vs Alert Volume")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(tradeoff_df["threshold"], tradeoff_df["alert_volume"])
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Alert Volume")
ax2.set_title("Threshold vs Alert Volume")
ax2.grid(True)
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(tradeoff_df["threshold"], tradeoff_df["recall"], label="Recall")
ax3.plot(tradeoff_df["threshold"], tradeoff_df["precision"], label="Precision")
ax3.set_xlabel("Threshold")
ax3.set_ylabel("Score")
ax3.set_title("Threshold vs Recall and Precision")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

# Explanation latency section
st.subheader("6. Explanation Latency Governance")

st.markdown(
    f"""
Assumed latency per explanation: **{latency_per_explanation_ms} ms**  
Current selected threshold produces **{metrics['alert_volume']:,} alerts**.  
Estimated total explanation time: **{metrics['total_explanation_latency_sec']:.2f} seconds**.

This demonstrates that explanation latency is not only a model-level issue. It also depends on alert volume.
"""
)

# Governance interpretation
st.subheader("7. Governance Interpretation")

if metrics["recall"] >= recall_target and metrics["alert_rate"] <= alert_rate_target:
    st.success("This threshold satisfies both recall and alert-volume constraints.")
elif metrics["recall"] >= recall_target:
    st.warning("This threshold satisfies recall, but alert volume may be operationally high.")
elif metrics["alert_rate"] <= alert_rate_target:
    st.warning("This threshold controls workload, but recall may be below the risk target.")
else:
    st.error("This threshold satisfies neither the recall target nor the alert-volume target.")

st.markdown(
    """
### Defence line
**The model does not define deployment behaviour by itself. The threshold governs how model scores become operational decisions.**
"""
)

# Download outputs
st.subheader("8. Export Results")

export_df = tradeoff_df.copy()
export_df["selected_manual_threshold"] = threshold

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download threshold trade-off CSV",
    data=csv,
    file_name="threshold_tradeoff_dashboard.csv",
    mime="text/csv",
)

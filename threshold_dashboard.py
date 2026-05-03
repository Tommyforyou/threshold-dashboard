# ============================================================
# ENHANCED GOVERNANCE THRESHOLD DASHBOARD — FULL PhD DEFENCE VERSION
# ============================================================
# Run:
#   streamlit run threshold_dashboard.py
#
# Install:
#   pip install streamlit pandas numpy scikit-learn matplotlib
#
# Optional for PDF export:
#   pip install reportlab
# ============================================================

import io
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
    page_title="SIDS Governance Feasibility Dashboard",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ SIDS Governance Feasibility Dashboard")

st.markdown(
    """
This dashboard demonstrates how fraud-detection thresholds affect **risk, workload, latency, explainability, and deployment feasibility**.

**Core principle:** the model produces scores, but governance determines how those scores become operational decisions.
"""
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("1. Data")
uploaded_file = st.sidebar.file_uploader("Upload creditcard.csv", type=["csv"])
max_rows = st.sidebar.slider("Maximum rows", 5_000, 100_000, 30_000, 5_000)
test_size = st.sidebar.slider("Test size", 0.10, 0.40, 0.20, 0.05)

st.sidebar.header("2. Threshold Control")
threshold = st.sidebar.slider("Manual threshold", 0.0, 1.0, 0.05, 0.001)
auto_play = st.sidebar.checkbox("Auto-play threshold movement", value=False)
auto_speed = st.sidebar.slider("Auto-play speed", 0.01, 0.20, 0.05, 0.01)

st.sidebar.header("3. Governance Constraints")
recall_target = st.sidebar.slider("Recall target", 0.50, 1.00, 0.90, 0.01)
alert_rate_target = st.sidebar.slider("Alert-rate target", 0.001, 0.100, 0.010, 0.001, format="%.3f")
latency_target_sec = st.sidebar.slider("Total explanation latency target (sec)", 1.0, 120.0, 30.0, 1.0)
latency_per_alert_ms = st.sidebar.slider("Latency per alert explanation (ms)", 1, 2000, 250, 10)
review_time_per_alert_min = st.sidebar.slider("Analyst review time per alert (min)", 1, 15, 3, 1)

st.sidebar.header("4. Scenario Presets")
scenario = st.sidebar.selectbox(
    "Scenario",
    [
        "Manual",
        "High-risk bank mode",
        "Normal operations mode",
        "Limited analyst capacity mode",
        "Regulatory audit mode",
    ],
)

if scenario == "High-risk bank mode":
    recall_target = 0.95
    alert_rate_target = 0.03
elif scenario == "Normal operations mode":
    recall_target = 0.90
    alert_rate_target = 0.02
elif scenario == "Limited analyst capacity mode":
    recall_target = 0.85
    alert_rate_target = 0.01
elif scenario == "Regulatory audit mode":
    recall_target = 0.95
    alert_rate_target = 0.05

# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=True)
def load_data(file, max_rows):
    df = pd.read_csv(file)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a target column named 'Class'.")
    if len(df) > max_rows:
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
# MODEL FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=True)
def train_models(df, test_size):
    df = add_sids_proxy_features(df)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                solver="liblinear",
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=60,
                max_depth=7,
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            )),
        ]),
    }

    trained = {}
    score_dict = {}
    training_times = {}
    scoring_times = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        training_times[name] = time.time() - start

        start = time.time()
        score_dict[name] = model.predict_proba(X_test)[:, 1]
        scoring_times[name] = time.time() - start
        trained[name] = model

    return trained, X_test, y_test, score_dict, training_times, scoring_times

# ============================================================
# METRICS
# ============================================================

def safe_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_metrics(y_true, y_scores, threshold, latency_per_alert_ms):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = safe_confusion_matrix(y_true, y_pred)
    alert_volume = int(y_pred.sum())
    alert_rate = float(y_pred.mean())
    total_latency_sec = (alert_volume * latency_per_alert_ms) / 1000

    return {
        "threshold": float(threshold),
        "auc_pr": float(average_precision_score(y_true, y_scores)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "alert_volume": alert_volume,
        "alert_rate": alert_rate,
        "total_explanation_latency_sec": float(total_latency_sec),
    }


def make_tradeoff_df(y_true, y_scores, latency_per_alert_ms):
    thresholds = np.unique(np.quantile(y_scores, np.linspace(0, 1, 140)))
    rows = [compute_metrics(y_true, y_scores, t, latency_per_alert_ms) for t in thresholds]
    return pd.DataFrame(rows)


def recommend_thresholds(tradeoff_df, recall_target, alert_rate_target, latency_target_sec):
    recommendations = {}

    # Best F1 threshold
    if len(tradeoff_df) > 0:
        recommendations["Best F1"] = tradeoff_df.loc[tradeoff_df["f1"].idxmax()].to_dict()

    # Recall-constrained: meet recall with lowest alert volume
    recall_df = tradeoff_df[tradeoff_df["recall"] >= recall_target]
    if len(recall_df) > 0:
        recommendations["Recall-based"] = recall_df.sort_values("alert_volume").iloc[0].to_dict()

    # Alert-volume-constrained: meet alert rate with highest recall
    alert_df = tradeoff_df[tradeoff_df["alert_rate"] <= alert_rate_target]
    if len(alert_df) > 0:
        recommendations["Alert-volume-based"] = alert_df.sort_values("recall", ascending=False).iloc[0].to_dict()

    # Full feasibility envelope: recall + alert + latency
    feasible_df = tradeoff_df[
        (tradeoff_df["recall"] >= recall_target)
        & (tradeoff_df["alert_rate"] <= alert_rate_target)
        & (tradeoff_df["total_explanation_latency_sec"] <= latency_target_sec)
    ]
    if len(feasible_df) > 0:
        recommendations["Feasibility Envelope"] = feasible_df.sort_values(["f1", "precision"], ascending=False).iloc[0].to_dict()

    return recommendations

# ============================================================
# EXPLANATION VIEW
# ============================================================

def generate_explanation(model, X_test, transaction_position):
    """Lightweight feature-contribution view.
    For Logistic Regression: coefficient × transformed value.
    For Random Forest: feature importance × raw absolute value, as a demonstration proxy.
    """
    X_row = X_test.iloc[[transaction_position]]
    final_model = model.named_steps["model"]

    if isinstance(final_model, LogisticRegression):
        preprocessor = model[:-1]
        transformed = preprocessor.transform(X_row)
        coefs = final_model.coef_[0]
        contributions = transformed[0] * coefs
    elif isinstance(final_model, RandomForestClassifier):
        importances = final_model.feature_importances_
        raw_values = X_row.iloc[0].fillna(0).values
        contributions = importances * np.abs(raw_values)
    else:
        contributions = np.zeros(len(X_test.columns))

    explanation_df = pd.DataFrame({
        "feature": X_test.columns,
        "original_value": X_row.iloc[0].values,
        "contribution": contributions,
        "absolute_contribution": np.abs(contributions),
    })

    return explanation_df.sort_values("absolute_contribution", ascending=False).head(10)

# ============================================================
# REPORT EXPORT
# ============================================================

def create_text_report(dataset_summary, selected_model, current_metrics, recommendations):
    lines = []
    lines.append("SIDS Governance Feasibility Report")
    lines.append("=" * 45)
    lines.append("")
    lines.append("Dataset Summary")
    lines.append(json.dumps(dataset_summary, indent=2))
    lines.append("")
    lines.append(f"Selected Model: {selected_model}")
    lines.append("")
    lines.append("Current Threshold Metrics")
    lines.append(json.dumps(current_metrics, indent=2))
    lines.append("")
    lines.append("Recommended Operating Points")
    lines.append(json.dumps(recommendations, indent=2))
    lines.append("")
    lines.append("Governance Interpretation")
    lines.append("The threshold governs how model scores become operational decisions. Feasibility depends on recall, alert volume, explanation latency, and analyst workload.")
    return "\n".join(lines)

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

fraud_count = int(df["Class"].sum())
total_count = int(len(df))
fraud_rate = float(df["Class"].mean())

st.subheader("1. Dataset Governance Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{total_count:,}")
c2.metric("Fraud cases", f"{fraud_count:,}")
c3.metric("Fraud rate", f"{fraud_rate:.4%}")
c4.metric("Missing values", f"{int(df.isnull().sum().sum()):,}")

dataset_summary = {
    "transactions": total_count,
    "fraud_cases": fraud_count,
    "legitimate_cases": total_count - fraud_count,
    "fraud_rate": fraud_rate,
    "missing_values": int(df.isnull().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
}

with st.spinner("Training models and scoring transactions..."):
    models, X_test, y_test, score_dict, training_times, scoring_times = train_models(df, test_size)

st.subheader("2. Model Comparison")

comparison_rows = []
for model_name, scores in score_dict.items():
    m = compute_metrics(y_test, scores, threshold, latency_per_alert_ms)
    comparison_rows.append({
        "model": model_name,
        "AUC-PR": m["auc_pr"],
        "ROC-AUC": m["roc_auc"],
        "Recall": m["recall"],
        "Precision": m["precision"],
        "F1": m["f1"],
        "Alerts": m["alert_volume"],
        "Training sec": training_times[model_name],
        "Scoring sec": scoring_times[model_name],
    })

comparison_df = pd.DataFrame(comparison_rows)
st.dataframe(comparison_df, use_container_width=True)

selected_model = st.selectbox("Select model for interactive governance analysis", list(models.keys()))
model = models[selected_model]
y_scores = score_dict[selected_model]

if auto_play:
    threshold = float((time.time() * auto_speed) % 1.0)

metrics = compute_metrics(y_test, y_scores, threshold, latency_per_alert_ms)
tradeoff_df = make_tradeoff_df(y_test, y_scores, latency_per_alert_ms)
recommendations = recommend_thresholds(tradeoff_df, recall_target, alert_rate_target, latency_target_sec)

st.subheader("3. Live Threshold Impact")
st.caption(f"Selected model: {selected_model} | Current threshold: {threshold:.4f}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Recall", f"{metrics['recall']:.3f}")
k2.metric("Precision", f"{metrics['precision']:.3f}")
k3.metric("Alert Volume", f"{metrics['alert_volume']:,}")
k4.metric("Alert Rate", f"{metrics['alert_rate']:.2%}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
k6.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
k7.metric("F1", f"{metrics['f1']:.3f}")
k8.metric("Explanation Latency", f"{metrics['total_explanation_latency_sec']:.2f} sec")

st.markdown("### Confusion Matrix")
cm_df = pd.DataFrame(
    {
        "Predicted Legitimate": [metrics["true_negatives"], metrics["false_negatives"]],
        "Predicted Fraud": [metrics["false_positives"], metrics["true_positives"]],
    },
    index=["Actual Legitimate", "Actual Fraud"],
)
st.dataframe(cm_df, use_container_width=True)

st.subheader("4. Governance Status Panel")

recall_pass = metrics["recall"] >= recall_target
alert_pass = metrics["alert_rate"] <= alert_rate_target
latency_pass = metrics["total_explanation_latency_sec"] <= latency_target_sec

s1, s2, s3, s4 = st.columns(4)
s1.metric("Recall Target", "PASS" if recall_pass else "FAIL", f"{metrics['recall']:.3f} / {recall_target:.2f}")
s2.metric("Alert-Rate Target", "PASS" if alert_pass else "FAIL", f"{metrics['alert_rate']:.2%} / {alert_rate_target:.2%}")
s3.metric("Latency Target", "PASS" if latency_pass else "FAIL", f"{metrics['total_explanation_latency_sec']:.2f}s / {latency_target_sec:.0f}s")
s4.metric("Deployment Status", "FEASIBLE" if recall_pass and alert_pass and latency_pass else "NOT FEASIBLE")

if recall_pass and alert_pass and latency_pass:
    st.success("Inside SIDS feasibility envelope ✅")
elif recall_pass and not alert_pass:
    st.warning("Recall is acceptable, but alert workload is too high.")
elif alert_pass and not recall_pass:
    st.warning("Workload is controlled, but recall is below risk target.")
else:
    st.error("Outside feasibility envelope ❌")

st.subheader("5. Threshold Recommendation Engine")

if recommendations:
    rec_table = pd.DataFrame(recommendations).T
    display_cols = ["threshold", "recall", "precision", "f1", "alert_volume", "alert_rate", "total_explanation_latency_sec"]
    st.dataframe(rec_table[display_cols], use_container_width=True)
else:
    st.warning("No recommended threshold found under current constraints.")

selected_recommendation = st.selectbox("Inspect recommended operating point", list(recommendations.keys()) if recommendations else ["None"])
if selected_recommendation != "None" and recommendations:
    st.json(recommendations[selected_recommendation])

st.subheader("6. Analyst Workload Simulation")

total_review_minutes = metrics["alert_volume"] * review_time_per_alert_min
analysts_needed = total_review_minutes / 480

w1, w2, w3 = st.columns(3)
w1.metric("Alerts", f"{metrics['alert_volume']:,}")
w2.metric("Total review time", f"{total_review_minutes:.0f} min")
w3.metric("Analysts required / 8h shift", f"{analysts_needed:.2f}")

st.subheader("7. Trade-Off Curves")

g1, g2 = st.columns(2)

with g1:
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(tradeoff_df["alert_volume"], tradeoff_df["recall"])
    ax1.scatter(metrics["alert_volume"], metrics["recall"], label="Current threshold")
    ax1.set_xlabel("Alert Volume")
    ax1.set_ylabel("Recall")
    ax1.set_title("Recall vs Alert Volume")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

with g2:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(tradeoff_df["threshold"], tradeoff_df["alert_volume"])
    ax2.scatter(metrics["threshold"], metrics["alert_volume"], label="Current threshold")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Alert Volume")
    ax2.set_title("Threshold vs Alert Volume")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.plot(tradeoff_df["threshold"], tradeoff_df["recall"], label="Recall")
ax3.plot(tradeoff_df["threshold"], tradeoff_df["precision"], label="Precision")
ax3.plot(tradeoff_df["threshold"], tradeoff_df["f1"], label="F1")
ax3.scatter(metrics["threshold"], metrics["recall"], label="Current recall")
ax3.set_xlabel("Threshold")
ax3.set_ylabel("Metric value")
ax3.set_title("Threshold vs Recall, Precision and F1")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

st.subheader("8. Transaction-Level Explanation View")

ranked_indices = np.argsort(y_scores)[::-1]
selected_rank = st.slider("Select transaction by risk rank", 1, min(100, len(ranked_indices)), 1, 1)
tx_pos = int(ranked_indices[selected_rank - 1])
tx_score = float(y_scores[tx_pos])
tx_decision = int(tx_score >= threshold)

st.markdown(
    f"""
**Risk rank:** {selected_rank}  
**Fraud probability:** {tx_score:.4f}  
**Current decision:** {'Fraud Alert' if tx_decision == 1 else 'Not Alerted'}
"""
)

explanation_df = generate_explanation(model, X_test, tx_pos)
st.dataframe(explanation_df[["feature", "original_value", "contribution"]], use_container_width=True)

fig4, ax4 = plt.subplots(figsize=(8, 4))
plot_df = explanation_df.sort_values("contribution")
ax4.barh(plot_df["feature"], plot_df["contribution"])
ax4.set_xlabel("Approximate feature contribution")
ax4.set_title("Top Feature Contributions")
ax4.grid(True)
st.pyplot(fig4)

st.subheader("9. Governance Interpretation")

st.markdown(
    """
**Defence line:** The model does not define deployment behaviour by itself. The threshold governs operational outcomes under real-world constraints.

This dashboard therefore operationalises explainability and threshold selection as a **governance mechanism**, not merely a technical tuning exercise.
"""
)

st.subheader("10. Export Regulator-Ready Outputs")

report_text = create_text_report(dataset_summary, selected_model, metrics, recommendations)

st.download_button(
    label="Download governance report (.txt)",
    data=report_text.encode("utf-8"),
    file_name="sids_governance_feasibility_report.txt",
    mime="text/plain",
)

csv = tradeoff_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download threshold trade-off table (.csv)",
    data=csv,
    file_name="threshold_tradeoff_table.csv",
    mime="text/csv",
)

comparison_csv = comparison_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download model comparison table (.csv)",
    data=comparison_csv,
    file_name="model_comparison_table.csv",
    mime="text/csv",
)

st.info("For formal PDF reports, export the text report and convert it to PDF, or extend this app with reportlab.")

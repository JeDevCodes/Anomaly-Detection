"""
Streamlit Dashboard
Interactive anomaly detection platform with 4 tabs:
  1. Dashboard  — run pipeline + overview
  2. Alerts     — prioritized alert queue
  3. Investigate — resolve alerts + feedback
  4. Performance — metrics + threshold tuning
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_ingestion import get_data
from src.feature_engine import engineer_features, get_feature_matrix
from src.detectors import run_all_detectors
from src.scorer import score_anomalies, get_score_summary, get_top_alerts
from src.alert_manager import (
    generate_alerts, deduplicate_alerts, get_alert_queue,
    get_alert_stats, save_alerts, adapt_thresholds,
)
from src.investigator import (
    update_status, resolve_alert, get_feedback_stats,
    get_feedback_by_type, empty_feedback_log, save_feedback,
)
import config


# Page Config 
st.set_page_config(
    page_title="Anomaly Detection System",
    layout="wide",
)


#  SESSION STATE INITIALIZATION

def init_state():
    """Initialize session state with defaults."""
    defaults = {
        "pipeline_run": False,
        "scored_df": None,
        "alerts": None,
        "feedback": empty_feedback_log(),
        "summary": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


#  SIDEBAR 

st.sidebar.title("🛡️ Anomaly Detection")
st.sidebar.markdown("---")

source = st.sidebar.radio("Data Source", ["Synthetic", "Upload CSV"])

n_rows = None
uploaded_file = None

if source == "Synthetic":
    n_rows = st.sidebar.slider("Number of Transactions", 1000, 20000, 5000, step=1000)
else:
    uploaded_file = st.sidebar.file_uploader("Upload Trading Data", type=["csv"])

run_btn = st.sidebar.button("🚀 Run Pipeline", use_container_width=True)


#  PIPELINE EXECUTION

if run_btn:
    with st.spinner("Running anomaly detection pipeline..."):
        # Data ingestion
        if source == "Upload CSV" and uploaded_file is not None:
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(uploaded_file.read())
            tmp.close()
            df = get_data(source="file", filepath=tmp.name)
            os.unlink(tmp.name)
        else:
            df = get_data(source="synthetic", n_rows=n_rows)

        # Features → Detect → Score
        df = engineer_features(df)
        fm = get_feature_matrix(df)
        results = run_all_detectors(fm)
        scored_df = score_anomalies(df, results)

        # Alerts
        alerts = generate_alerts(scored_df)
        alerts = deduplicate_alerts(alerts)

        # Save to session
        st.session_state.scored_df = scored_df
        st.session_state.alerts = alerts
        st.session_state.feedback = empty_feedback_log()
        st.session_state.summary = get_score_summary(scored_df)
        st.session_state.pipeline_run = True

    st.sidebar.success(f"✅ Processed {len(scored_df)} transactions")


#  TABS

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔔 Alerts", "🔍 Investigate", "📈 Performance"])


#  TAB 1 — DASHBOARD

with tab1:
    st.header("System Overview")

    if not st.session_state.pipeline_run:
        st.info("👈 Select data source and click **Run Pipeline** to start.")
        st.stop()

    summary = st.session_state.summary
    scored_df = st.session_state.scored_df

    # Metric Cards 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", summary["total_transactions"])
    c2.metric("Flagged", summary["total_flagged"], f"{summary['flag_rate_pct']}%")
    c3.metric("Avg Score", summary["score_avg"])
    c4.metric("Max Score", summary["score_max"])

    # Score Distribution 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution")
        fig = px.histogram(
            scored_df, x="anomaly_score", nbins=50,
            color="is_fraud", barmode="overlay",
            labels={"anomaly_score": "Anomaly Score", "is_fraud": "Is Fraud"},
            color_discrete_map={True: "#EF4444", False: "#3B82F6"},
        )
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Level Breakdown")
        risk_data = pd.DataFrame([
            {"level": level, "count": summary["by_risk_level"].get(level, 0)}
            for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
        ])
        colors = {"CRITICAL": "#EF4444", "HIGH": "#F97316", "MEDIUM": "#EAB308",
                  "LOW": "#22C55E", "NONE": "#94A3B8"}
        fig = px.bar(
            risk_data, x="level", y="count", color="level",
            color_discrete_map=colors,
        )
        fig.update_layout(height=350, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Fraud Type and Timeline
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Predicted Fraud Types")
        if summary["by_fraud_type"]:
            type_data = pd.DataFrame([
                {"type": k, "count": v}
                for k, v in summary["by_fraud_type"].items()
            ])
            fig = px.pie(type_data, names="type", values="count", hole=0.4)
            fig.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No anomalies detected.")

    with col4:
        st.subheader("Anomalies Over Time")
        flagged = scored_df[scored_df["anomaly_score"] >= config.ALERT_THRESHOLDS["LOW"]].copy()
        if len(flagged) > 0:
            flagged["date"] = pd.to_datetime(flagged["timestamp"]).dt.date
            daily = flagged.groupby("date").size().reset_index(name="count")
            fig = px.line(daily, x="date", y="count", markers=True)
            fig.update_layout(height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No anomalies to plot.")


#  TAB 2 — ALERT QUEUE

with tab2:
    st.header("Alert Queue")

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline first.")
        st.stop()

    alerts = st.session_state.alerts

    if len(alerts) == 0:
        st.success("No alerts generated. All transactions appear normal.")
        st.stop()

    # Filters 
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        level_filter = st.selectbox(
            "Risk Level", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
        )
    with fc2:
        type_filter = st.selectbox(
            "Fraud Type", ["ALL"] + list(alerts["predicted_fraud_type"].unique())
        )
    with fc3:
        status_filter = st.selectbox(
            "Status", ["ALL"] + list(alerts["status"].unique())
        )

    # Apply filters
    filtered = alerts.copy()
    if level_filter != "ALL":
        filtered = filtered[filtered["risk_level"] == level_filter]
    if type_filter != "ALL":
        filtered = filtered[filtered["predicted_fraud_type"] == type_filter]
    if status_filter != "ALL":
        filtered = filtered[filtered["status"] == status_filter]

    queue = get_alert_queue(filtered)

    st.write(f"Showing **{len(queue)}** of {len(alerts)} alerts")

    # Alert Table 
    display_cols = [
        "alert_id", "entity_id", "anomaly_score", "risk_level",
        "confidence", "predicted_fraud_type", "status",
    ]
    available_cols = [c for c in display_cols if c in queue.columns]
    st.dataframe(
        queue[available_cols],
        use_container_width=True,
        height=400,
    )

    # Alert Detail 
    # st.subheader("Alert Detail")
    # if len(queue) > 0:
    #     selected_id = st.selectbox("Select Alert", queue["alert_id"].tolist())
    #     selected = queue[queue["alert_id"] == selected_id].iloc[0]

    #     dc1, dc2 = st.columns(2)
    #     with dc1:
    #         st.markdown(f"**Alert ID:** {selected['alert_id']}")
    #         st.markdown(f"**Entity:** {selected['entity_id']}")
    #         st.markdown(f"**Score:** {selected['anomaly_score']}")
    #         st.markdown(f"**Risk Level:** {selected['risk_level']}")
    #     with dc2:
    #         st.markdown(f"**Fraud Type:** {selected['predicted_fraud_type']}")
    #         st.markdown(f"**Confidence:** {selected['confidence']}/3 detectors")
    #         st.markdown(f"**Status:** {selected['status']}")

        # st.markdown(f"**Explanation:** {selected['explanation']}")


#  TAB 3 — INVESTIGATION

with tab3:
    st.header("Investigation Workflow")

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline first.")
        st.stop()

    alerts = st.session_state.alerts

    #  Pending alerts (not yet resolved) 
    pending = alerts[alerts["status"] != "RESOLVED"]
    st.write(f"**Pending alerts:** {len(pending)} | **Resolved:** {len(alerts) - len(pending)}")

    if len(pending) == 0:
        st.success("All alerts have been resolved!")
    else:
        st.subheader("Resolve Alert")

        # Select alert to investigate
        pending_sorted = get_alert_queue(pending)
        selected_id = st.selectbox(
            "Select Alert to Investigate",
            pending_sorted["alert_id"].tolist(),
            key="investigate_select",
        )
        selected = pending_sorted[pending_sorted["alert_id"] == selected_id].iloc[0]

        # Show alert details
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | **Entity** | {selected['entity_id']} |
        | **Score** | {selected['anomaly_score']} ({selected['risk_level']}) |
        | **Type** | {selected['predicted_fraud_type']} |
        | **Confidence** | {selected['confidence']}/3 detectors |
        """)
        # st.markdown(f"**Explanation:** {selected['explanation']}")

        # Resolution form
        rc1, rc2 = st.columns(2)
        with rc1:
            resolution = st.selectbox("Resolution", config.RESOLUTION_TYPES)
        with rc2:
            notes = st.text_input("Investigation Notes", placeholder="Describe findings...")

        if st.button("✅ Resolve Alert", use_container_width=True):
            alerts, feedback = resolve_alert(
                st.session_state.alerts,
                st.session_state.feedback,
                alert_id=selected_id,
                resolution=resolution,
                notes=notes,
            )
            st.session_state.alerts = alerts
            st.session_state.feedback = feedback

            save_alerts(alerts)
            save_feedback(feedback)

            st.success(f"Alert {selected_id} resolved as {resolution}")
            st.rerun()

    #  Feedback Log 
    st.subheader("Resolution History")
    feedback = st.session_state.feedback
    if len(feedback) > 0:
        st.dataframe(feedback, use_container_width=True, height=250)
    else:
        st.write("No resolutions yet.")


#  TAB 4 — PERFORMANCE

with tab4:
    st.header("Detection Performance")

    if not st.session_state.pipeline_run:
        st.info("Run the pipeline first.")
        st.stop()

    scored_df = st.session_state.scored_df
    feedback = st.session_state.feedback

    # Ground Truth Metrics (if synthetic data) 
    if "is_fraud" in scored_df.columns and scored_df["is_fraud"].sum() > 0:
        st.subheader("Detection Accuracy (Ground Truth)")

        fraud_mask = scored_df["is_fraud"]
        flagged_mask = scored_df["anomaly_score"] >= config.ALERT_THRESHOLDS["LOW"]

        tp = int((flagged_mask & fraud_mask).sum())
        fp = int((flagged_mask & ~fraud_mask).sum())
        fn = int((~flagged_mask & fraud_mask).sum())
        tn = int((~flagged_mask & ~fraud_mask).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Precision", f"{precision:.3f}")
        mc2.metric("Recall", f"{recall:.3f}")
        mc3.metric("F1 Score", f"{f1:.3f}")
        mc4.metric("False Positive Rate", f"{fp / max(fp + tn, 1):.3f}")

        # Confusion matrix
        conf_data = pd.DataFrame(
            [[tp, fp], [fn, tn]],
            index=["Actually Fraud", "Actually Normal"],
            columns=["Flagged", "Not Flagged"],
        )
        st.dataframe(conf_data, use_container_width=True)

        # Score separation chart
        # st.subheader("Score Separation: Fraud vs Normal")
        # fig = go.Figure()
        # fig.add_trace(go.Box(
        #     y=scored_df.loc[~fraud_mask, "anomaly_score"],
        #     name="Normal", marker_color="#3B82F6",
        # ))
        # fig.add_trace(go.Box(
        #     y=scored_df.loc[fraud_mask, "anomaly_score"],
        #     name="Fraud", marker_color="#EF4444",
        # ))
        # fig.update_layout(height=350, margin=dict(t=20, b=20))
        # st.plotly_chart(fig, use_container_width=True)

    #  Feedback-Based Metrics 
    st.subheader("Investigation Feedback Metrics")
    if len(feedback) > 0:
        stats = get_feedback_stats(feedback)

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Resolved", stats["total_resolved"])
        fc2.metric("True Positives", stats["true_positives"])
        fc3.metric("False Positives", stats["false_positives"])
        fc4.metric("FP Rate", f"{stats['fp_rate']:.1%}")

        # Per-type breakdown
        by_type = get_feedback_by_type(feedback)
        if by_type:
            st.subheader("Performance by Fraud Type")
            type_rows = []
            for ftype, fstats in by_type.items():
                type_rows.append({
                    "Fraud Type": ftype,
                    "TP": fstats["true_positives"],
                    "FP": fstats["false_positives"],
                    "Escalated": fstats["escalated"],
                    "FP Rate": f"{fstats['fp_rate']:.1%}",
                })
            st.dataframe(pd.DataFrame(type_rows), use_container_width=True)

        # Adaptive thresholds
        st.subheader("Adaptive Thresholds")
        st.write(f"Current: {config.ALERT_THRESHOLDS}")
        new_thresholds = adapt_thresholds(stats)
        if new_thresholds != config.ALERT_THRESHOLDS:
            st.warning(f"Recommended adjustment: {new_thresholds}")
        else:
            st.success("Thresholds are well-calibrated. No adjustment needed.")
    else:
        st.info("Resolve some alerts in the Investigation tab to see feedback metrics.")
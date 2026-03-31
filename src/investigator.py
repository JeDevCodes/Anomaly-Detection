"""
Investigation Workflow Module
Manages alert lifecycle: status updates, resolutions, and feedback loop.
Feedback stats feed back into alert_manager.adapt_thresholds() to
reduce false positives over time.
"""

import pandas as pd
from datetime import datetime
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


#  STATUS MANAGEMENT

def update_status(alerts, alert_id, new_status):
    """
    Move an alert to the next status in the workflow.

    Args:
        alerts: alerts DataFrame
        alert_id: which alert to update
        new_status: one of config.ALERT_STATUSES

    Returns:
        Updated alerts DataFrame
    """
    alerts = alerts.copy()

    if new_status not in config.ALERT_STATUSES:
        raise ValueError(f"Invalid status: {new_status}. Use: {config.ALERT_STATUSES}")

    mask = alerts["alert_id"] == alert_id
    if mask.sum() == 0:
        raise ValueError(f"Alert not found: {alert_id}")

    alerts.loc[mask, "status"] = new_status
    return alerts


def bulk_update_status(alerts, alert_ids, new_status):
    """
    Update multiple alerts to same status at once.

    Args:
        alerts: alerts DataFrame
        alert_ids: list of alert IDs
        new_status: target status

    Returns:
        Updated alerts DataFrame
    """
    alerts = alerts.copy()

    if new_status not in config.ALERT_STATUSES:
        raise ValueError(f"Invalid status: {new_status}")

    mask = alerts["alert_id"].isin(alert_ids)
    alerts.loc[mask, "status"] = new_status
    return alerts

#  RESOLUTION

def resolve_alert(alerts, feedback_log, alert_id, resolution, notes=""):
    """
    Resolve an alert and record the decision in feedback log.

    Args:
        alerts: alerts DataFrame
        feedback_log: existing feedback DataFrame (or empty)
        alert_id: which alert to resolve
        resolution: TRUE_POSITIVE / FALSE_POSITIVE / ESCALATED
        notes: investigator's notes

    Returns:
        (updated_alerts, updated_feedback_log)
    """
    if resolution not in config.RESOLUTION_TYPES:
        raise ValueError(f"Invalid resolution: {resolution}. Use: {config.RESOLUTION_TYPES}")

    # Update alert status
    alerts = update_status(alerts, alert_id, "RESOLVED")

    # Get alert details for feedback record
    alert_row = alerts[alerts["alert_id"] == alert_id].iloc[0]

    # Create feedback entry
    entry = pd.DataFrame([{
        "alert_id": alert_id,
        "entity_id": alert_row["entity_id"],
        "anomaly_score": alert_row["anomaly_score"],
        "risk_level": alert_row["risk_level"],
        "predicted_fraud_type": alert_row["predicted_fraud_type"],
        "resolution": resolution,
        "notes": notes,
        "resolved_at": datetime.now(),
    }])

    # Append to feedback log
    feedback_log = pd.concat([feedback_log, entry], ignore_index=True)

    return alerts, feedback_log


def batch_resolve(alerts, feedback_log, resolutions):
    """
    Resolve multiple alerts at once.

    Args:
        alerts: alerts DataFrame
        feedback_log: existing feedback DataFrame
        resolutions: list of dicts with keys: alert_id, resolution, notes

    Returns:
        (updated_alerts, updated_feedback_log)
    """
    for r in resolutions:
        alerts, feedback_log = resolve_alert(
            alerts, feedback_log,
            alert_id=r["alert_id"],
            resolution=r["resolution"],
            notes=r.get("notes", ""),
        )
    return alerts, feedback_log


#  FEEDBACK STATISTICS

def get_feedback_stats(feedback_log):
    """
    Compute investigation performance metrics from resolved alerts.
    Output feeds directly into alert_manager.adapt_thresholds().

    Args:
        feedback_log: DataFrame of resolved alerts

    Returns:
        dict with fp_rate, miss_rate, precision, recall, total_resolved
    """
    if len(feedback_log) == 0:
        return {
            "fp_rate": 0.0,
            "miss_rate": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "total_resolved": 0,
            "true_positives": 0,
            "false_positives": 0,
            "escalated": 0,
        }

    tp = (feedback_log["resolution"] == "TRUE_POSITIVE").sum()
    fp = (feedback_log["resolution"] == "FALSE_POSITIVE").sum()
    esc = (feedback_log["resolution"] == "ESCALATED").sum()
    total = len(feedback_log)

    # FP rate: what fraction of our alerts were wrong
    fp_rate = fp / max(total, 1)

    # Precision: of all alerts we raised, how many were real
    precision = tp / max(tp + fp, 1)

    # Miss rate approximation: escalated cases suggest potential misses
    miss_rate = esc / max(total, 1)

    return {
        "fp_rate": round(fp_rate, 4),
        "miss_rate": round(miss_rate, 4),
        "precision": round(precision, 4),
        "recall": round(1.0 - miss_rate, 4),
        "total_resolved": int(total),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "escalated": int(esc),
    }


def get_feedback_by_type(feedback_log):
    """
    Break down feedback stats by predicted fraud type.
    Shows which fraud types have highest false positive rates.

    Args:
        feedback_log: DataFrame of resolved alerts

    Returns:
        dict mapping fraud_type → {tp, fp, esc, fp_rate}
    """
    if len(feedback_log) == 0:
        return {}

    result = {}
    for ftype in feedback_log["predicted_fraud_type"].unique():
        subset = feedback_log[feedback_log["predicted_fraud_type"] == ftype]
        tp = (subset["resolution"] == "TRUE_POSITIVE").sum()
        fp = (subset["resolution"] == "FALSE_POSITIVE").sum()
        esc = (subset["resolution"] == "ESCALATED").sum()

        result[ftype] = {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "escalated": int(esc),
            "total": int(len(subset)),
            "fp_rate": round(fp / max(len(subset), 1), 4),
            "miss_rate": round(esc / max(len(subset), 1), 4),   
        }

    return result


#  PERSISTENCE

def empty_feedback_log():
    """Create an empty feedback log DataFrame."""
    return pd.DataFrame(columns=[
        "alert_id", "entity_id", "anomaly_score", "risk_level",
        "predicted_fraud_type", "resolution", "notes", "resolved_at",
    ])


def save_feedback(feedback_log, filepath=None):
    """Save feedback log to CSV."""
    filepath = filepath or config.FEEDBACK_FILE
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    feedback_log.to_csv(filepath, index=False)
    return filepath


def load_feedback(filepath=None):
    """Load feedback log from CSV."""
    filepath = filepath or config.FEEDBACK_FILE
    if not os.path.exists(filepath):
        return empty_feedback_log()
    return pd.read_csv(filepath)


# ═══════════════════════════════════════════════
#  SELF TEST
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    from data_ingestion import get_data
    from feature_engine import engineer_features, get_feature_matrix
    from detectors import run_all_detectors
    from scorer import score_anomalies
    from alert_manager import generate_alerts, deduplicate_alerts, get_alert_queue, adapt_thresholds

    print("=" * 60)
    print("  INVESTIGATOR — SELF TEST")
    print("=" * 60)

    # ── Full pipeline up to alerts ──
    df = get_data(source="synthetic", n_rows=2000)
    df = engineer_features(df)
    fm = get_feature_matrix(df)
    results = run_all_detectors(fm)
    scored_df = score_anomalies(df, results)
    alerts = generate_alerts(scored_df)
    alerts = deduplicate_alerts(alerts)
    queue = get_alert_queue(alerts)
    print(f"\nAlerts in queue: {len(queue)}")

    # ── Simulate investigation workflow ──
    feedback = empty_feedback_log()

    # Take top 20 alerts and resolve them
    top_20 = queue.head(20)
    print(f"\n{'─' * 60}")
    print("  SIMULATING INVESTIGATION")

    for i, (_, alert) in enumerate(top_20.iterrows()):
        # Simulate: high score alerts are usually TP, low score often FP
        if alert["anomaly_score"] >= 70:
            resolution = "TRUE_POSITIVE"
        elif alert["anomaly_score"] >= 50:
            resolution = "ESCALATED" if i % 3 == 0 else "TRUE_POSITIVE"
        else:
            resolution = "FALSE_POSITIVE" if i % 2 == 0 else "TRUE_POSITIVE"

        # Step through workflow
        alerts = update_status(alerts, alert["alert_id"], "REVIEWING")
        alerts = update_status(alerts, alert["alert_id"], "INVESTIGATED")
        alerts, feedback = resolve_alert(
            alerts, feedback,
            alert_id=alert["alert_id"],
            resolution=resolution,
            notes=f"Simulated investigation #{i+1}",
        )

    # ── Check statuses ──
    print(f"\n  Status breakdown:")
    print(f"  {alerts['status'].value_counts().to_dict()}")

    # ── Feedback stats ──
    stats = get_feedback_stats(feedback)
    print(f"\n{'─' * 60}")
    print("  FEEDBACK STATISTICS")
    for key, val in stats.items():
        print(f"    {key}: {val}")

    # ── Per-type breakdown ──
    by_type = get_feedback_by_type(feedback)
    print(f"\n  By fraud type:")
    for ftype, fstats in by_type.items():
        print(f"    {ftype}: {fstats}")

    # ── Adaptive threshold feedback loop ──
    print(f"\n{'─' * 60}")
    print("  ADAPTIVE THRESHOLD LOOP")
    print(f"  Current thresholds: {config.ALERT_THRESHOLDS}")
    new_thresholds = adapt_thresholds(stats)
    print(f"  Adapted thresholds: {new_thresholds}")
    changed = {k: new_thresholds[k] - config.ALERT_THRESHOLDS[k]
               for k in new_thresholds if new_thresholds[k] != config.ALERT_THRESHOLDS[k]}
    if changed:
        print(f"  Changes: {changed}")
    else:
        print(f"  No changes needed (FP rate acceptable)")

    # ── Persistence ──
    path = save_feedback(feedback)
    loaded = load_feedback(path)
    assert len(loaded) == len(feedback), "Save/load mismatch!"
    print(f"\n  Saved {len(feedback)} feedback records → {path}")
    print(f"  ✓ Persistence verified")

    print(f"\n{'=' * 60}")
    print("  SELF TEST COMPLETE ✓")
    print(f"{'=' * 60}")        
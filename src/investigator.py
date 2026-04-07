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


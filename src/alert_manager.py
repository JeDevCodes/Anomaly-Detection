"""
Alert Manager Module
Generates, deduplicates, prioritizes, and persists anomaly alerts.
Supports adaptive thresholds that learn from false positive feedback.
"""

import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Risk level sort order (highest priority first)
RISK_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

#  ALERT GENERATION

def generate_alerts(scored_df):
    """
    Create alert records from scored transactions.
    Only transactions with score >= LOW threshold become alerts.

    Args:
        scored_df: DataFrame from score_anomalies()

    Returns:
        DataFrame of alert records
    """
    min_threshold = config.ALERT_THRESHOLDS["LOW"]
    flagged = scored_df[scored_df["anomaly_score"] >= min_threshold].copy()

    if len(flagged) == 0:
        return _empty_alerts()

    alerts = pd.DataFrame({
        "alert_id": [f"ALT-{i:06d}" for i in range(len(flagged))],
        "trade_ids": flagged["trade_id"].values,
        "entity_id": flagged["entity_id"].values,
        "symbol": flagged["symbol"].values,
        "timestamp": flagged["timestamp"].values,
        "anomaly_score": flagged["anomaly_score"].values,
        "risk_level": flagged["risk_level"].values,
        "confidence": flagged["confidence"].values,
        "predicted_fraud_type": flagged["predicted_fraud_type"].values,
        # "explanation": flagged["explanation"].values,
        "status": "NEW",
        "created_at": datetime.now(),
    })

    return alerts.reset_index(drop=True)


def _empty_alerts():
    """Return empty DataFrame with correct alert schema."""
    cols = [
        "alert_id", "trade_ids", "entity_id", "symbol", "timestamp",
        "anomaly_score", "risk_level", "confidence", "predicted_fraud_type",
        "explanation", "status", "created_at",
    ]
    return pd.DataFrame(columns=cols)


#  DEDUPLICATION

def deduplicate_alerts(alerts):
    """
    Merge alerts from same entity + same fraud type within time window.
    Keeps highest score, combines trade_ids.

    Args:
        alerts: DataFrame from generate_alerts()

    Returns:
        Deduplicated alerts DataFrame
    """
    if len(alerts) == 0:
        return alerts

    alerts = alerts.copy()
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"])

    # Sort by entity, fraud type, then time
    alerts = alerts.sort_values(["entity_id", "predicted_fraud_type", "timestamp"])

    window = pd.Timedelta(minutes=config.DEDUP_WINDOW_MINUTES)
    groups = []
    current_group = None

    for _, row in alerts.iterrows():
        if current_group is None:
            current_group = _new_group(row)
            continue

        # Same entity + same fraud type + within time window?
        same_entity = row["entity_id"] == current_group["entity_id"]
        same_type = row["predicted_fraud_type"] == current_group["predicted_fraud_type"]
        within_window = (row["timestamp"] - current_group["last_timestamp"]) <= window

        if same_entity and same_type and within_window:
            # Merge into current group
            current_group["trade_ids"].append(row["trade_ids"])
            current_group["anomaly_score"] = max(current_group["anomaly_score"], row["anomaly_score"])
            current_group["confidence"] = max(current_group["confidence"], row["confidence"])
            current_group["last_timestamp"] = row["timestamp"]
            current_group["count"] += 1
        else:
            groups.append(current_group)
            current_group = _new_group(row)

    if current_group:
        groups.append(current_group)

    # Build deduplicated DataFrame
    deduped = pd.DataFrame([{
        "alert_id": f"ALT-{i:06d}",
        "trade_ids": ",".join(g["trade_ids"]),
        "entity_id": g["entity_id"],
        "symbol": g["symbol"],
        "timestamp": g["first_timestamp"],
        "anomaly_score": g["anomaly_score"],
        "risk_level": _score_to_level(g["anomaly_score"]),
        "confidence": g["confidence"],
        "predicted_fraud_type": g["predicted_fraud_type"],
        "explanation": g["explanation"] + (
            f" ({g['count']} related trades merged.)" if g["count"] > 1 else ""
        ),
        "trade_count": g["count"],
        "status": "NEW",
        "created_at": datetime.now(),
    } for i, g in enumerate(groups)])

    return deduped.reset_index(drop=True)


def _new_group(row):
    """Initialize a new dedup group from an alert row."""
    return {
        "entity_id": row["entity_id"],
        "symbol": row["symbol"],
        "predicted_fraud_type": row["predicted_fraud_type"],
        "trade_ids": [row["trade_ids"]],
        "anomaly_score": row["anomaly_score"],
        "confidence": row["confidence"],
        "explanation": row["explanation"],
        "first_timestamp": row["timestamp"],
        "last_timestamp": row["timestamp"],
        "count": 1,
    }


def _score_to_level(score):
    """Convert score back to risk level after dedup score update."""
    thresholds = config.ALERT_THRESHOLDS
    if score >= thresholds["CRITICAL"]:
        return "CRITICAL"
    if score >= thresholds["HIGH"]:
        return "HIGH"
    if score >= thresholds["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# ═══════════════════════════════════════════════
#  PRIORITIZATION
# ═══════════════════════════════════════════════

def get_alert_queue(alerts, risk_level=None, status=None):
    """
    Return sorted alert queue. Highest priority first.

    Args:
        alerts: alerts DataFrame
        risk_level: filter to specific level (optional)
        status: filter to specific status (optional)

    Returns:
        Sorted DataFrame
    """
    if len(alerts) == 0:
        return alerts

    filtered = alerts.copy()

    if risk_level:
        filtered = filtered[filtered["risk_level"] == risk_level]

    if status:
        filtered = filtered[filtered["status"] == status]

    # Sort: risk rank first, then score descending
    filtered["_rank"] = filtered["risk_level"].map(RISK_RANK).fillna(99)
    filtered = filtered.sort_values(
        ["_rank", "anomaly_score"],
        ascending=[True, False]
    ).drop(columns=["_rank"])

    return filtered.reset_index(drop=True)


def get_alert_stats(alerts):
    """
    Summary statistics for the alert queue.

    Args:
        alerts: alerts DataFrame

    Returns:
        dict with counts and breakdowns
    """
    if len(alerts) == 0:
        return {"total": 0, "by_level": {}, "by_status": {}, "by_type": {}}

    return {
        "total": len(alerts),
        "by_level": alerts["risk_level"].value_counts().to_dict(),
        "by_status": alerts["status"].value_counts().to_dict(),
        "by_type": alerts["predicted_fraud_type"].value_counts().to_dict(),
        "avg_score": round(alerts["anomaly_score"].mean(), 1),
    }


# ═══════════════════════════════════════════════
#  ADAPTIVE THRESHOLDS
# ═══════════════════════════════════════════════

def adapt_thresholds(feedback_stats):
    """
    Adjust alert thresholds based on investigation feedback.
    If too many false positives → raise thresholds (fewer alerts).
    If missing too many real threats → lower thresholds (more alerts).

    Args:
        feedback_stats: dict with keys "fp_rate" and "miss_rate" (0.0 to 1.0)

    Returns:
        dict of updated thresholds
    """
    thresholds = config.ALERT_THRESHOLDS.copy()
    adjustment = config.THRESHOLD_ADJUSTMENT

    fp_rate = feedback_stats.get("fp_rate", 0.0)
    miss_rate = feedback_stats.get("miss_rate", 0.0)

    if fp_rate > config.FP_RATE_UPPER:
        # Too many false positives → raise all thresholds
        for level in thresholds:
            thresholds[level] = min(thresholds[level] + adjustment, 99)

    elif miss_rate > config.MISS_RATE_UPPER:
        # Missing real threats → lower all thresholds
        for level in thresholds:
            thresholds[level] = max(thresholds[level] - adjustment, 5)

    return thresholds


# ═══════════════════════════════════════════════
#  PERSISTENCE
# ═══════════════════════════════════════════════

def save_alerts(alerts, filepath=None):
    """Save alerts to CSV file."""
    filepath = filepath or config.ALERT_LOG_FILE
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    alerts.to_csv(filepath, index=False)
    return filepath


def load_alerts(filepath=None):
    """Load alerts from CSV file."""
    filepath = filepath or config.ALERT_LOG_FILE
    if not os.path.exists(filepath):
        return _empty_alerts()
    return pd.read_csv(filepath)


# ═══════════════════════════════════════════════
#  SELF TEST
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    from data_ingestion import get_data
    from feature_engine import engineer_features, get_feature_matrix
    from detectors import run_all_detectors
    from scorer import score_anomalies

    print("=" * 60)
    print("  ALERT MANAGER — SELF TEST")
    print("=" * 60)

    # ── Full pipeline ──
    df = get_data(source="synthetic", n_rows=2000)
    df = engineer_features(df)
    fm = get_feature_matrix(df)
    results = run_all_detectors(fm)
    scored_df = score_anomalies(df, results)

    # ── Generate alerts ──
    alerts = generate_alerts(scored_df)
    print(f"\nRaw alerts generated: {len(alerts)}")

    # ── Deduplicate ──
    deduped = deduplicate_alerts(alerts)
    print(f"After deduplication: {len(deduped)}")
    print(f"Reduction: {len(alerts) - len(deduped)} alerts merged")

    # ── Priority queue ──
    queue = get_alert_queue(deduped)
    print(f"\n{'─' * 60}")
    print("  ALERT QUEUE (top 10)")
    print(f"  {'Rank':<5} {'Alert':<12} {'Entity':<12} {'Score':>6} {'Level':<10} {'Type':<18}")
    print(f"  {'─' * 65}")
    for i, (_, row) in enumerate(queue.head(10).iterrows(), 1):
        print(f"  {i:<5} {row['alert_id']:<12} {row['entity_id']:<12} "
              f"{row['anomaly_score']:>5.1f} {row['risk_level']:<10} "
              f"{row['predicted_fraud_type']:<18}")

    # ── Stats ──
    stats = get_alert_stats(deduped)
    print(f"\n{'─' * 60}")
    print("  ALERT STATISTICS")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    # ── Filter by level ──
    critical = get_alert_queue(deduped, risk_level="CRITICAL")
    high = get_alert_queue(deduped, risk_level="HIGH")
    print(f"\n  Critical alerts: {len(critical)}")
    print(f"  High alerts: {len(high)}")

    # ── Adaptive thresholds ──
    print(f"\n{'─' * 60}")
    print("  ADAPTIVE THRESHOLDS")
    print(f"  Current: {config.ALERT_THRESHOLDS}")

    high_fp = adapt_thresholds({"fp_rate": 0.40, "miss_rate": 0.01})
    print(f"  After high FP (40%): {high_fp}")

    high_miss = adapt_thresholds({"fp_rate": 0.10, "miss_rate": 0.10})
    print(f"  After high miss (10%): {high_miss}")

    # ── Persistence ──
    path = save_alerts(deduped)
    loaded = load_alerts(path)
    print(f"\n  Saved {len(deduped)} alerts → {path}")
    print(f"  Loaded back: {len(loaded)} alerts")
    assert len(loaded) == len(deduped), "Save/load mismatch!"
    print("  ✓ Persistence verified")

    print(f"\n{'=' * 60}")
    print("  SELF TEST COMPLETE ✓")
    print(f"{'=' * 60}")
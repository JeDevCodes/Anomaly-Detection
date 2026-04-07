"""
Anomaly Scorer Module
Combines all detector outputs into a unified 0-100 anomaly score.
Provides:
  - Weighted ensemble scoring with consensus boost
  - Risk level assignment (CRITICAL / HIGH / MEDIUM / LOW / NONE)
  - Fraud type prediction via feature signature matching
  - Human-readable explanations with feature attribution
  - System-level summary statistics
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engine import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS


# Columns added by this module
SCORE_COLUMNS = [
    "anomaly_score",
    "risk_level",
    "confidence",
    "predicted_fraud_type",
]


def score_anomalies(df, detector_results):
    """
    Main entry point. Enriches DataFrame with anomaly scores and explanations.

    Args:
        df: DataFrame with feature columns (output of engineer_features)
        detector_results: dict from run_all_detectors()

    Returns:
        DataFrame with 5 new scoring columns added
    """
    df = df.copy()
    combined = detector_results["combined"]

    # Step 1 — Weighted ensemble score (0-100)
    scores = _compute_ensemble_score(combined)
    df["anomaly_score"] = scores

    # Step 2 — Risk level
    df["risk_level"] = _assign_risk_levels(scores)

    # Step 3 — Detector agreement count
    df["confidence"] = combined["n_flags"].values

    # Step 4 — Predict fraud pattern
    df["predicted_fraud_type"] = _predict_fraud_types(df, scores)

    return df


def get_score_summary(scored_df):
    """
    System-level summary statistics for dashboard and monitoring.

    Args:
        scored_df: DataFrame output from score_anomalies()

    Returns:
        dict with counts, distributions, and averages
    """
    total = len(scored_df)
    min_threshold = config.ALERT_THRESHOLDS["LOW"]
    flagged_df = scored_df[scored_df["anomaly_score"] >= min_threshold]
    n_flagged = len(flagged_df)

    summary = {
        "total_transactions": total,
        "total_flagged": n_flagged,
        "flag_rate_pct": round(n_flagged / max(total, 1) * 100, 2),
        "by_risk_level": scored_df["risk_level"].value_counts().to_dict(),
        "by_fraud_type": (
            flagged_df["predicted_fraud_type"].value_counts().to_dict()
            if n_flagged > 0 else {}
        ),
        "score_avg": round(scored_df["anomaly_score"].mean(), 2),
        "score_median": round(scored_df["anomaly_score"].median(), 2),
        "score_max": round(scored_df["anomaly_score"].max(), 2),
        "flagged_avg_score": (
            round(flagged_df["anomaly_score"].mean(), 2)
            if n_flagged > 0 else 0
        ),
        "detector_agreement": (
            flagged_df["confidence"].value_counts().sort_index().to_dict()
            if n_flagged > 0 else {}
        ),
        # genrates the agreement between the detectors in the flagged transactions
        # e.g. how many flagged transactions had 1, 2, or all 3 detectors flagging them {1: 10, 2: 25, 3: 15}
    }

    return summary


def get_top_alerts(scored_df, n=10):
    """
    Return the top N highest-scoring transactions.

    Args:
        scored_df: DataFrame output from score_anomalies()
        n: number of alerts to return

    Returns:
        DataFrame sorted by anomaly_score descending
    """
    min_threshold = config.ALERT_THRESHOLDS["LOW"]
    flagged = scored_df[scored_df["anomaly_score"] >= min_threshold]
    return flagged.sort_values("anomaly_score", ascending=False).head(n)


#  ENSEMBLE SCORING

def _compute_ensemble_score(combined):
    """
    Weighted combination of detector scores → 0-100.
    Applies consensus boost when multiple detectors agree.

    Args:
        combined: DataFrame with stat_score, if_score, lof_score, n_flags

    Returns:
        numpy array of scores (0-100)
    """
    weights = config.DETECTOR_WEIGHTS

    raw = (
        combined["stat_score"].values * weights["statistical"]
        + combined["if_score"].values * weights["isolation_forest"]
        + combined["lof_score"].values * weights["lof"]
    )

    # Scale to 0-100
    scores = raw * 100.0

    # Consensus boost: reward multi-detector agreement
    n_flags = combined["n_flags"].values
    boost = np.where(
        n_flags >= 3, 1.15,
        np.where(n_flags >= 2, 1.08, 1.0)
    )
    scores = scores * boost

    return np.round(scores.clip(0, 100), 1)


#  RISK LEVEL ASSIGNMENT

def _assign_risk_levels(scores):
    """
    Map scores to risk level strings using config thresholds.
    Applied in ascending order so higher levels overwrite lower.

    Args:
        scores: numpy array of 0-100 scores

    Returns:
        numpy array of risk level strings
    """
    thresholds = config.ALERT_THRESHOLDS
    levels = np.full(len(scores), "NONE", dtype=object)

    levels[scores >= thresholds["LOW"]] = "LOW"
    levels[scores >= thresholds["MEDIUM"]] = "MEDIUM"
    levels[scores >= thresholds["HIGH"]] = "HIGH"
    levels[scores >= thresholds["CRITICAL"]] = "CRITICAL"

    return levels


#  FRAUD TYPE PREDICTION
def _predict_fraud_types(df, scores):
    """
    Predict most likely fraud pattern via feature signature matching.
    Fully vectorized — no row-level loops.

    Each fraud type has a signature: weighted sum of relevant features.
    The fraud type with the highest signature score wins.

    Args:
        df: DataFrame with feature columns
        scores: numpy array of anomaly scores

    Returns:
        pandas Series of predicted fraud type strings
    """
    sigs = pd.DataFrame(index=df.index)

    # Wash trading: entity trades with itself + high volume
    sigs["wash_trading"] = (
        df["is_self_trade"] * 5.0
        + df["quantity_zscore"].clip(lower=0) * 1.0
        + (1.0 - df["price_vs_symbol_avg"].abs()).clip(lower=0) * 0.5
    )

    # Spoofing: high cancel rate + large orders
    sigs["spoofing"] = (
        (df["entity_cancel_rate"] - 0.1).clip(lower=0) * 6.0
        + df["quantity_zscore"].clip(lower=0) * 1.0
    )

    # Pump & dump: rapid price movement + volume surge
    sigs["pump_and_dump"] = (
        df["price_velocity"].abs() * 3.0
        + df["symbol_volume_zscore"].clip(lower=0) * 2.0
        + df["price_vs_symbol_avg"].abs() * 1.0
    )

    # Insider trading: sudden high frequency + oversized trades
    sigs["insider_trading"] = (
        (df["entity_trade_frequency"] - 5).clip(lower=0) * 2.0
        + (df["quantity_vs_entity_avg"] - 2).clip(lower=0) * 2.5
        + (60 - df["time_since_last_trade"]).clip(lower=0) * 0.02
    )

    # Layering: cancel rate + rapid succession + volatility
    sigs["layering"] = (
        (df["entity_cancel_rate"] - 0.1).clip(lower=0) * 4.0
        + (30 - df["time_since_last_trade"]).clip(lower=0) * 0.05
        + df["symbol_volatility"].clip(lower=0) * 0.5
    )

    predicted = sigs.idxmax(axis=1)
    max_sig = sigs.max(axis=1)

    # Only assign type if signature score is meaningful AND anomaly score qualifies
    predicted[max_sig < 0.5] = "unknown"
    predicted[scores < config.ALERT_THRESHOLDS["LOW"]] = "none"

    return predicted

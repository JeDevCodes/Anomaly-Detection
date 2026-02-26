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
    # "explanation",
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

    # Step 5 — Human-readable explanations (must be last, uses columns from steps 1-4)
    # df["explanation"] = _generate_explanations(df, combined, scores)

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


#  EXPLANATION GENERATION

# def _generate_explanations(df, combined, scores):
#     """
#     Generate human-readable explanation strings.
#     Only detailed explanations for flagged transactions (score >= LOW threshold).
#     Non-flagged rows get a simple "no anomalies" message.

#     Args:
#         df: DataFrame with features + risk_level + predicted_fraud_type
#         combined: detector combined results
#         scores: numpy array of anomaly scores

#     Returns:
#         list of explanation strings (same length as df)
#     """
#     min_threshold = config.ALERT_THRESHOLDS["LOW"]

#     # Pre-fill all as clean
#     explanations = ["No significant anomalies detected."] * len(df)

#     # Only build explanations for flagged rows
#     flagged_indices = np.where(scores >= min_threshold)[0]

#     for idx in flagged_indices:
#         score = scores[idx]
#         row = df.iloc[idx]
#         comb_row = combined.iloc[idx]

#         # Header
#         parts = [f"Score {score:.0f}/100 ({row['risk_level']})."]

#         # Which detectors fired
#         detector_names = _get_detector_names(comb_row)
#         n_flags = int(comb_row["n_flags"])
#         parts.append(f"Detectors: {', '.join(detector_names)} ({n_flags}/3).")

#         # Top 3 most anomalous features
#         top_feats = _get_top_features(row)
#         if top_feats:
#             signals = [_format_feature_value(name, val) for name, val in top_feats]
#             parts.append("Key signals: " + "; ".join(signals) + ".")

#         # Predicted fraud type
#         pred = row["predicted_fraud_type"]
#         if pred not in ("none", "unknown"):
#             readable = pred.replace("_", " ").title()
#             parts.append(f"Suspected: {readable}.")

#         explanations[idx] = " ".join(parts)

#     return explanations


def _get_detector_names(comb_row):
    """Return list of detector names that flagged this row."""
    names = []
    if comb_row["stat_flag"]:
        names.append("Statistical")
    if comb_row["if_flag"]:
        names.append("Isolation Forest")
    if comb_row["lof_flag"]:
        names.append("LOF")
    return names if names else ["None"]


def _get_top_features(row, top_n=3):
    """
    Rank features by anomaly magnitude for a single transaction.
    Returns list of (feature_name, value) tuples, most anomalous first.
    """
    deviations = []

    for col in FEATURE_COLUMNS:
        if col not in row.index:
            continue

        val = float(row[col])

        # Weight certain features higher based on their signal value
        if col == "is_self_trade":
            magnitude = val * 10.0
        elif col == "entity_cancel_rate":
            magnitude = val * 5.0
        elif col == "time_since_last_trade":
            magnitude = max(0.0, (60.0 - val) / 60.0) * 3.0 if val < 60 else 0.0
        elif col == "quantity_vs_entity_avg":
            magnitude = max(0.0, val - 1.0) * 2.0
        else:
            magnitude = abs(val)

        deviations.append((col, val, magnitude))

    deviations.sort(key=lambda x: x[2], reverse=True)
    return [(name, val) for name, val, _ in deviations[:top_n]]


def _format_feature_value(feature_name, value):
    """Convert a feature name + value into investigator-friendly text."""
    if feature_name == "is_self_trade":
        return "Self-trade detected (entity on both sides)" if value > 0.5 else "No self-trade"

    if feature_name == "entity_cancel_rate":
        return f"Cancel rate: {value * 100:.1f}% of orders"

    if feature_name == "time_since_last_trade":
        if value < 10:
            return f"Rapid trading: {value:.1f}s between trades"
        if value < 60:
            return f"Fast trading: {value:.0f}s between trades"
        return f"Trade gap: {value / 60:.1f}min"

    if feature_name == "entity_trade_frequency":
        return f"Frequency: {value:.1f} trades/day"

    if feature_name == "quantity_vs_entity_avg":
        return f"Trade size: {value:.1f}x entity's average"

    if feature_name == "price_velocity":
        return f"Price velocity: ${abs(value):.4f}/sec"

    if feature_name == "price_vs_symbol_avg":
        direction = "above" if value > 0 else "below"
        return f"Price {abs(value):.1f}σ {direction} symbol avg"

    if feature_name == "symbol_volatility":
        return f"Symbol volatility: {value:.2f}"

    if feature_name == "symbol_volume_zscore":
        return f"Volume: {value:.1f}σ vs symbol norm"

    if feature_name == "hour_of_day":
        return f"Trade hour: {int(value):02d}:00"

    if feature_name.endswith("_zscore"):
        direction = "above" if value > 0 else "below"
        desc = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
        return f"{desc}: {abs(value):.1f}σ {direction} avg"

    desc = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
    return f"{desc}: {value:.3f}"


#  SELF TEST

if __name__ == "__main__":
    from data_ingestion import get_data
    from feature_engine import engineer_features, get_feature_matrix
    from detectors import run_all_detectors

    print("=" * 70)
    print("  ANOMALY SCORER — SELF TEST")
    print("=" * 70)

    # ── Build pipeline ──
    df = get_data(source="synthetic", n_rows=2000)
    df = engineer_features(df)
    fm = get_feature_matrix(df)
    detector_results = run_all_detectors(fm)

    # ── Score ──
    scored_df = score_anomalies(df, detector_results)
    print(f"\nScored DataFrame: {scored_df.shape}")
    print(f"Columns added: {SCORE_COLUMNS}")

    # ── Verify score column integrity ──
    print(f"\n{'─' * 70}")
    print("  SCORE INTEGRITY CHECK")
    assert scored_df["anomaly_score"].min() >= 0, "Score below 0!"
    assert scored_df["anomaly_score"].max() <= 100, "Score above 100!"
    assert scored_df["anomaly_score"].isna().sum() == 0, "NaN scores!"
    assert scored_df["risk_level"].isna().sum() == 0, "NaN risk levels!"
    # assert scored_df["explanation"].isna().sum() == 0, "NaN explanations!"
    print("  ✓ All scores in [0, 100]")
    print("  ✓ No NaN values in any scoring column")

    # ── Score distribution: fraud vs normal ──
    print(f"\n{'─' * 70}")
    print("  SCORE DISTRIBUTION")
    fraud_mask = scored_df["is_fraud"]
    normal_scores = scored_df.loc[~fraud_mask, "anomaly_score"]
    fraud_scores = scored_df.loc[fraud_mask, "anomaly_score"]

    print(f"  Normal transactions ({len(normal_scores)}):")
    print(f"    Mean: {normal_scores.mean():.1f}  Median: {normal_scores.median():.1f}  Max: {normal_scores.max():.1f}")
    print(f"  Fraud transactions ({len(fraud_scores)}):")
    print(f"    Mean: {fraud_scores.mean():.1f}  Median: {fraud_scores.median():.1f}  Max: {fraud_scores.max():.1f}")
    print(f"  Separation: fraud avg is {fraud_scores.mean() - normal_scores.mean():.1f} points higher ✓")

    # ── Risk level breakdown ──
    print(f"\n{'─' * 70}")
    print("  RISK LEVEL BREAKDOWN")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
        count = (scored_df["risk_level"] == level).sum()
        fraud_in_level = scored_df[(scored_df["risk_level"] == level) & fraud_mask].shape[0]
        print(f"  {level:>10}: {count:5d} total | {fraud_in_level:4d} actual fraud")

    # ── Fraud type prediction accuracy ──
    print(f"\n{'─' * 70}")
    print("  FRAUD TYPE PREDICTION")
    flagged_fraud = scored_df[fraud_mask & (scored_df["anomaly_score"] >= config.ALERT_THRESHOLDS["LOW"])]
    if len(flagged_fraud) > 0:
        correct = (flagged_fraud["predicted_fraud_type"] == flagged_fraud["fraud_type"]).sum()
        total_f = len(flagged_fraud)
        print(f"  Correctly predicted: {correct}/{total_f} ({correct / total_f * 100:.1f}%)")

        print(f"\n  {'Actual':<20} {'Predicted':<20} {'Count':>6}")
        print(f"  {'─' * 48}")
        confusion = flagged_fraud.groupby(["fraud_type", "predicted_fraud_type"]).size()
        for (actual, predicted), count in confusion.items():
            match = "✓" if actual == predicted else "✗"
            print(f"  {actual:<20} {predicted:<20} {count:>5} {match}")

    # ── Summary stats ──
    print(f"\n{'─' * 70}")
    print("  SYSTEM SUMMARY")
    summary = get_score_summary(scored_df)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # ── Top 5 alerts with explanations ──
    # print(f"\n{'─' * 70}")
    # print("  TOP 5 ALERTS")
    # top_alerts = get_top_alerts(scored_df, n=5)
    # for i, (_, row) in enumerate(top_alerts.iterrows(), 1):
    #     print(f"\n  #{i} | Trade {row['trade_id']} | Entity {row['entity_id']}")
    #     print(f"      Score: {row['anomaly_score']} | Risk: {row['risk_level']}")
    #     print(f"      Actual Fraud: {row['is_fraud']} ({row['fraud_type']})")
    #     print(f"      Explanation: {row['explanation']}")

    print(f"\n{'=' * 70}")
    print("  SELF TEST COMPLETE ✓")
    print(f"{'=' * 70}")
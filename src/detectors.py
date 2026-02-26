"""
Anomaly Detection Engine
Runs 5 detectors (3 statistical + 2 ML) on the feature matrix.
Each returns a standardized result dict.
run_all_detectors() orchestrates everything and returns combined results.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.feature_engine import FEATURE_COLUMNS


def run_all_detectors(feature_matrix):
    """
    Run all 5 detectors and return structured results.

    Args:
        feature_matrix: DataFrame with FEATURE_COLUMNS (from get_feature_matrix)

    Returns:
        dict with keys:
            "statistical"       → combined stat detector result
            "isolation_forest"  → IF result
            "lof"               → LOF result
            "combined"          → DataFrame with per-row scores from each method
    """
    fm = feature_matrix.copy()
    n = len(fm)

    # Standardize features once — shared by all detectors
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(fm),
        columns=fm.columns,
        index=fm.index,
    )

    # Run each detector 
    stat_result = _run_statistical_detectors(fm, scaled)
    if_result = _run_isolation_forest(scaled)
    lof_result = _run_lof(scaled)

    # Build combined results DataFrame
    combined = pd.DataFrame(index=fm.index)
    combined["stat_score"] = stat_result["scores"]
    combined["if_score"] = if_result["scores"]
    combined["lof_score"] = lof_result["scores"]
    combined["stat_flag"] = stat_result["labels"]
    combined["if_flag"] = if_result["labels"]
    combined["lof_flag"] = lof_result["labels"]

    # Number of detectors that flagged each row (0-3)
    combined["n_flags"] = (
        combined["stat_flag"].astype(int)
        + combined["if_flag"].astype(int)
        + combined["lof_flag"].astype(int)
    )

    return {
        "statistical": stat_result,
        "isolation_forest": if_result,
        "lof": lof_result,
        "combined": combined,
    }

#  STATISTICAL DETECTORS

def _run_statistical_detectors(raw_features, scaled_features):
    """
    Runs Z-Score, IQR, and MAD detectors.
    Combines them into a single statistical score per row.

    Returns standardized result dict.
    """
    n = len(raw_features)

    zscore_flags = _zscore_detector(scaled_features)
    iqr_flags = _iqr_detector(raw_features)
    mad_flags = _mad_detector(raw_features)

    # Score = fraction of (detector × feature) combinations that flagged
    # 3 detectors × 12 features = 36 total checks
    total_flags = zscore_flags + iqr_flags + mad_flags  # each is (n, 12)
    max_possible = 3 * len(FEATURE_COLUMNS)

    scores = total_flags.sum(axis=1).values / max_possible  # 0 to 1
    scores = scores.clip(0, 1)

    # Flag if score exceeds a reasonable threshold
    threshold = 0.15  # flagged by >15% of checks
    labels = scores >= threshold

    # Feature importances: which features contributed most
    feature_importance = total_flags.sum(axis=0) / (total_flags.sum() + 1e-10)
    importances = dict(zip(FEATURE_COLUMNS, feature_importance.values))

    return {
        "method": "statistical",
        "scores": scores,
        "labels": labels,
        "threshold": threshold,
        "feature_importances": importances,
    }


def _zscore_detector(scaled_features):
    """
    Flag features where |z-score| > threshold.
    Returns DataFrame of booleans (n_rows × n_features).
    """
    threshold = config.ZSCORE_THRESHOLD
    return (scaled_features.abs() > threshold).astype(int)


def _iqr_detector(raw_features):
    """
    Flag features outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Returns DataFrame of booleans (n_rows × n_features).
    """
    q1 = raw_features.quantile(0.25)
    q3 = raw_features.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    flags = ((raw_features < lower) | (raw_features > upper)).astype(int)
    return flags


def _mad_detector(raw_features):
    """
    Median Absolute Deviation detector.
    Flag if deviation from median > 3 * MAD.
    More robust than z-score when outliers already exist.
    Returns DataFrame of booleans (n_rows × n_features).
    """
    median = raw_features.median()
    abs_deviation = (raw_features - median).abs()
    mad = abs_deviation.median()

    # Avoid division by zero — replace 0 MAD with 1
    mad_safe = mad.replace(0, 1.0)

    modified_zscore = abs_deviation / mad_safe
    flags = (modified_zscore > 3.0).astype(int)
    return flags


#  ML-BASED DETECTORS

def _run_isolation_forest(scaled_features):
    """
    Isolation Forest detector.
    Isolates anomalies by random partitioning — anomalies need fewer splits.

    Returns standardized result dict.
    """
    model = IsolationForest(
        n_estimators=config.IF_N_ESTIMATORS,
        contamination=config.IF_CONTAMINATION,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(scaled_features)

    # decision_function: lower = more anomalous (negative = anomaly)
    raw_scores = model.decision_function(scaled_features)

    # Convert to 0-1 scale: most anomalous = 1, most normal = 0
    scores = _normalize_scores(-raw_scores)

    # Predict labels: -1 = anomaly, 1 = normal
    predictions = model.predict(scaled_features)
    labels = predictions == -1

    # Feature importances via mean path length contribution
    importances = _compute_if_importances(model, scaled_features)

    return {
        "method": "isolation_forest",
        "scores": scores,
        "labels": labels,
        "threshold": config.IF_CONTAMINATION,
        "feature_importances": importances,
        "model": model,
    }


def _run_lof(scaled_features):
    """
    Local Outlier Factor detector.
    Compares local density of each point to its neighbors.
    Points in low-density regions are flagged as anomalies.

    Returns standardized result dict.
    """
    model = LocalOutlierFactor(
        n_neighbors=min(config.LOF_N_NEIGHBORS, len(scaled_features) - 1),
        contamination=config.LOF_CONTAMINATION,
        novelty=False,
        n_jobs=-1,
    )

    predictions = model.fit_predict(scaled_features)

    # negative_outlier_factor_: lower (more negative) = more anomalous
    raw_scores = model.negative_outlier_factor_

    # Convert to 0-1 scale
    scores = _normalize_scores(-raw_scores)

    labels = predictions == -1

    # Feature importances approximation via per-feature LOF contribution
    importances = _compute_lof_importances(scaled_features, labels)

    return {
        "method": "lof",
        "scores": scores,
        "labels": labels,
        "threshold": config.LOF_CONTAMINATION,
        "feature_importances": importances,
    }
   

#  UTILITY FUNCTIONS

def _normalize_scores(raw_scores):
    """
    Min-max normalize any score array to [0, 1].
    Handles edge case where all values are the same.
    """
    min_val = raw_scores.min()
    max_val = raw_scores.max()
    range_val = max_val - min_val

    if range_val == 0:
        return np.zeros_like(raw_scores, dtype=float)

    return (raw_scores - min_val) / range_val


def _compute_if_importances(model, scaled_features):
    """
    Approximate feature importances for Isolation Forest.
    Measures how much each feature contributes to anomaly scores
    by checking score variance when each feature is shuffled.
    """
    base_scores = -model.decision_function(scaled_features)
    importances = {}

    for col in scaled_features.columns:
        shuffled = scaled_features.copy()
        shuffled[col] = np.random.permutation(shuffled[col].values)
        new_scores = -model.decision_function(shuffled)
        # Higher difference = feature was more important
        importances[col] = float(np.abs(base_scores - new_scores).mean())

    # Normalize to sum to 1
    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}

    return importances


def _compute_lof_importances(scaled_features, labels):
    """
    Approximate feature importances for LOF.
    Compares mean feature values of anomalies vs normal points.
    Higher difference = feature is more discriminative.
    """
    importances = {}

    if labels.sum() == 0:
        return {col: 1.0 / len(FEATURE_COLUMNS) for col in FEATURE_COLUMNS}

    for col in scaled_features.columns:
        normal_mean = scaled_features.loc[~labels, col].mean()
        anomaly_mean = scaled_features.loc[labels, col].mean()
        importances[col] = float(abs(anomaly_mean - normal_mean))

    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}

    return importances


#  SELF TEST

if __name__ == "__main__":
    from data_ingestion import get_data
    from feature_engine import engineer_features, get_feature_matrix

    print("=" * 70)
    print("  DETECTOR ENGINE — SELF TEST")
    print("=" * 70)

    # Generate data and features
    df = get_data(source="synthetic", n_rows=2000)
    df = engineer_features(df)
    fm = get_feature_matrix(df)
    print(f"\nFeature matrix: {fm.shape}")

    # Run all detectors
    results = run_all_detectors(fm)

    # ── Print per-detector results ──
    for method in ["statistical", "isolation_forest", "lof"]:
        r = results[method]
        n_flagged = r["labels"].sum()
        avg_score = r["scores"].mean()
        print(f"\n{'─' * 50}")
        print(f"  {r['method'].upper()}")
        print(f"  Flagged: {n_flagged}/{len(r['labels'])} ({n_flagged/len(r['labels'])*100:.1f}%)")
        print(f"  Avg Score: {avg_score:.4f}")
        print(f"  Top 3 Important Features:")
        sorted_imp = sorted(r["feature_importances"].items(), key=lambda x: x[1], reverse=True)
        for fname, fval in sorted_imp[:3]:
            print(f"    {fname:<30} {fval:.4f}")

    # ── Combined results ──
    combined = results["combined"]
    print(f"\n{'─' * 50}")
    print(f"  COMBINED RESULTS")
    print(f"  Rows flagged by 1+ detectors: {(combined['n_flags'] >= 1).sum()}")
    print(f"  Rows flagged by 2+ detectors: {(combined['n_flags'] >= 2).sum()}")
    print(f"  Rows flagged by ALL 3:        {(combined['n_flags'] == 3).sum()}")

    # ── Compare against ground truth ──
    fraud_mask = df["is_fraud"].values
    n_fraud = fraud_mask.sum()
    n_normal = (~fraud_mask).sum()

    print(f"\n{'─' * 50}")
    print(f"  GROUND TRUTH COMPARISON")
    print(f"  Total fraud: {n_fraud} | Total normal: {n_normal}")

    for method in ["statistical", "isolation_forest", "lof"]:
        labels = results[method]["labels"]
        tp = (labels & fraud_mask).sum()
        fp = (labels & ~fraud_mask).sum()
        fn = (~labels & fraud_mask).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"\n  {method.upper()}")
        print(f"    True Positives:  {tp}")
        print(f"    False Positives: {fp}")
        print(f"    Precision: {precision:.3f} | Recall: {recall:.3f}")

    # Multi-detector consensus
    for threshold in [1, 2, 3]:
        consensus = combined["n_flags"] >= threshold
        tp = (consensus & fraud_mask).sum()
        fp = (consensus & ~fraud_mask).sum()
        fn = (~consensus & fraud_mask).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"\n  CONSENSUS ({threshold}+ detectors)")
        print(f"    TP: {tp} | FP: {fp} | Precision: {precision:.3f} | Recall: {recall:.3f}")

    print(f"\n{'=' * 70}")
    print("  SELF TEST COMPLETE")
    print(f"{'=' * 70}")
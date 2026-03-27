"""
CLI Entry Point
Runs the full anomaly detection pipeline from command line.

Usage:
  python main.py                                             # synthetic data, default settings
  python main.py --rows 5000                                 # synthetic with 5000 rows
  python main.py --source file --filepath data/trades.csv    # external data
"""

import argparse
import sys
import os

from src.data_ingestion import get_data
from src.feature_engine import engineer_features, get_feature_matrix
from src.detectors import run_all_detectors
from src.scorer import score_anomalies, get_score_summary, get_top_alerts
from src.alert_manager import (
    generate_alerts, deduplicate_alerts, get_alert_queue,
    get_alert_stats, save_alerts,
)
from src.investigator import empty_feedback_log, save_feedback
import config


def run_pipeline(source="synthetic", filepath=None, n_rows=None):
    """
    Execute the full detection pipeline and return all results.

    Returns:
        dict with keys: df, scored_df, alerts, queue, summary, stats
    """
    # Step 1 — Data
    print("\n[1/6] Loading data...")
    df = get_data(source=source, filepath=filepath, n_rows=n_rows)

    # Step 2 — Features
    print("[2/6] Engineering features...")
    df = engineer_features(df)
    fm = get_feature_matrix(df)

    # Step 3 — Detection
    print("[3/6] Running detectors...")
    detector_results = run_all_detectors(fm)

    # Step 4 — Scoring
    print("[4/6] Scoring anomalies...")
    scored_df = score_anomalies(df, detector_results)

    # Step 5 — Alerts
    print("[5/6] Generating alerts...")
    alerts = generate_alerts(scored_df)
    alerts = deduplicate_alerts(alerts)
    queue = get_alert_queue(alerts)

    # Step 6 — Save
    print("[6/6] Saving results...")
    save_alerts(alerts)
    save_feedback(empty_feedback_log())

    summary = get_score_summary(scored_df)
    stats = get_alert_stats(alerts)

    return {
        "df": df,
        "scored_df": scored_df,
        "detector_results": detector_results,
        "alerts": alerts,
        "queue": queue,
        "summary": summary,
        "stats": stats,
    }


def print_report(results):
    """Print a formatted pipeline report to console."""
    summary = results["summary"]
    stats = results["stats"]
    queue = results["queue"]
    scored_df = results["scored_df"]

    print("\n" + "=" * 65)
    print("  ANOMALY DETECTION SYSTEM — PIPELINE REPORT")
    print("=" * 65)

    # Overview
    print(f"\n  Transactions processed: {summary['total_transactions']}")
    print(f"  Anomalies flagged:     {summary['total_flagged']} ({summary['flag_rate_pct']}%)")
    print(f"  Average score:         {summary['score_avg']}")
    print(f"  Max score:             {summary['score_max']}")

    # Risk breakdown
    print(f"\n{'─' * 65}")
    print("  RISK LEVEL BREAKDOWN")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
        count = summary["by_risk_level"].get(level, 0)
        bar = "█" * min(count, 40)
        print(f"    {level:>10}: {count:5d}  {bar}")

    # Fraud type breakdown
    if summary["by_fraud_type"]:
        print(f"\n{'─' * 65}")
        print("  PREDICTED FRAUD TYPES")
        for ftype, count in sorted(summary["by_fraud_type"].items(), key=lambda x: -x[1]):
            print(f"    {ftype:<20}: {count}")

    # Ground truth comparison (if available)
    if "is_fraud" in scored_df.columns:
        fraud_mask = scored_df["is_fraud"]
        n_fraud = fraud_mask.sum()
        if n_fraud > 0:
            flagged_mask = scored_df["anomaly_score"] >= config.ALERT_THRESHOLDS["LOW"]
            tp = (flagged_mask & fraud_mask).sum()
            fp = (flagged_mask & ~fraud_mask).sum()
            fn = (~flagged_mask & fraud_mask).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)

            print(f"\n{'─' * 65}")
            print("  DETECTION PERFORMANCE (vs ground truth)")
            print(f"    True Positives:  {tp}")
            print(f"    False Positives: {fp}")
            print(f"    Missed (FN):     {fn}")
            print(f"    Precision:       {precision:.3f}")
            print(f"    Recall:          {recall:.3f}")
            print(f"    F1 Score:        {2 * precision * recall / max(precision + recall, 0.001):.3f}")

    # Top alerts
    top = queue.head(5)
    if len(top) > 0:
        print(f"\n{'─' * 65}")
        print("  TOP 5 ALERTS")
        for i, (_, row) in enumerate(top.iterrows(), 1):
            print(f"\n  #{i}  {row['alert_id']} | {row['entity_id']} | "
                  f"Score: {row['anomaly_score']} | {row['risk_level']}")
            print(f"      {row['explanation'][:100]}...")

    print(f"\n{'─' * 65}")
    print(f"  Alerts saved to: {config.ALERT_LOG_FILE}")
    print(f"  Run 'streamlit run app.py' for interactive investigation")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Market Anomaly Detection System")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "file"],
                        help="Data source: synthetic or file")
    parser.add_argument("--filepath", default=None,
                        help="Path to CSV/JSON file (required if source=file)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Number of rows for synthetic data")
    args = parser.parse_args()

    if args.source == "file" and not args.filepath:
        parser.error("--filepath required when --source=file")

    results = run_pipeline(
        source=args.source,
        filepath=args.filepath,
        n_rows=args.rows,
    )

    print_report(results)


if __name__ == "__main__":
    main()
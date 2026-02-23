# Anomaly Detection
Market anomaly detection platform for fraud and manipulation identification. Implement statistical and ML-based anomaly scoring, alert prioritization, and investigation workflows. Create pattern recognition for unusual trading activity and false positive management.

# Market Anomaly Detection Platform

A fraud and market manipulation detection system built from the perspective of a Chief Security Officer at a major financial institution.

## What It Does

- **Ingests** trading data (external CSV/JSON or synthetic generation)
- **Engineers** 12 behavioral features per transaction
- **Detects** anomalies using statistical (Z-Score, IQR, MAD) and ML (Isolation Forest, LOF) methods
- **Scores** each transaction 0–100 with human-readable explanations
- **Prioritizes** alerts (Critical / High / Medium / Low) with adaptive thresholds
- **Manages** investigations with a feedback loop that reduces false positives over time

## Fraud Patterns Detected

| Pattern | Description |
|---|---|
| Wash Trading | Self-trades inflating volume |
| Spoofing | Large orders placed then rapidly cancelled |
| Pump & Dump | Artificial price inflation then sell-off |
| Insider Trading | Unusual pre-announcement activity |
| Layering | Multiple deceptive order levels |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI pipeline
python main.py

# Run dashboard
streamlit run app.py

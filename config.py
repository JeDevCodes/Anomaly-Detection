"""
Anomaly Detection System — Central Configuration
All thresholds, weights, and parameters in one place.
"""

# ── Data Settings ──
SYNTHETIC_ROWS = 10000
FRAUD_INJECTION_RATE = 0.05  # 5% of transactions are fraudulent
RANDOM_SEED = 42

# ── Feature Engineering ──
ROLLING_WINDOW = 20  # rolling window for entity-level features
ZSCORE_THRESHOLD = 3.0

# ── Detector Weights (must sum to 1.0) ──
DETECTOR_WEIGHTS = {
    "statistical": 0.3,
    "isolation_forest": 0.4,
    "lof": 0.3,
}

# ── Isolation Forest ──
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS = 100

# ── LOF ──
LOF_N_NEIGHBORS = 20
LOF_CONTAMINATION = 0.05

# ── Alert Thresholds ──
ALERT_THRESHOLDS = {
    "CRITICAL": 90,
    "HIGH": 70,
    "MEDIUM": 50,
    "LOW": 30,
}

# ── Adaptive Threshold Settings ──
FP_RATE_UPPER = 0.30   # if FP rate > 30%, raise thresholds
MISS_RATE_UPPER = 0.05  # if miss rate > 5%, lower thresholds
THRESHOLD_ADJUSTMENT = 5  # points to shift

# ── Alert Deduplication ──
DEDUP_WINDOW_MINUTES = 60

# ── Investigation Statuses ──
ALERT_STATUSES = ["NEW", "REVIEWING", "INVESTIGATED", "RESOLVED"]
RESOLUTION_TYPES = ["TRUE_POSITIVE", "FALSE_POSITIVE", "ESCALATED"]

# ── Fraud Pattern Labels ──
FRAUD_TYPES = [
    "wash_trading",
    "spoofing",
    "pump_and_dump",
    "insider_trading",
    "layering",
]

# ── File Paths ──
DATA_DIR = "data"
FEEDBACK_FILE = "data/feedback_log.csv"
ALERT_LOG_FILE = "data/alert_log.csv"
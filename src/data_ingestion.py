import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Required columns for external data validation ──
REQUIRED_COLUMNS = ["timestamp", "entity_id", "symbol", "trade_type", "price", "quantity"]
OPTIONAL_DEFAULTS = {
    "trade_id": None,
    "amount": None,
    "is_cancelled": False,
    "counterparty": "unknown",
    "is_fraud": False,
    "fraud_type": None,
}

SYMBOLS = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "META", "NVDA", "JPM"]


def generate_synthetic_data(n_rows=None, seed=None):
    """
    Generate synthetic trading data with realistic patterns
    and inject fraud across 5 pattern types.
    
    Returns:
        pd.DataFrame with unified schema
    """
    n_rows = n_rows or config.SYNTHETIC_ROWS
    seed = seed or config.RANDOM_SEED
    np.random.seed(seed)

    n_fraud = int(n_rows * config.FRAUD_INJECTION_RATE)
    n_normal = n_rows - n_fraud

    # ── Generate normal trades ──
    normal_df = _generate_normal_trades(n_normal, seed)

    # ── Inject fraud patterns ──
    fraud_df = _generate_fraud_trades(n_fraud, seed)

    # ── Combine and shuffle ──
    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Assign trade IDs ──
    df["trade_id"] = [f"TXN-{i:06d}" for i in range(len(df))]

    # ── Calculate amount ──
    df["amount"] = df["price"] * df["quantity"]

    # ── Sort by timestamp ──
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def _generate_normal_trades(n, seed):
    """Generate normal/legitimate trading activity."""
    np.random.seed(seed)

    base_time = datetime(2024, 1, 15, 9, 30, 0)
    n_entities = 200

    records = []
    for i in range(n):
        # spread trades across a trading day (6.5 hours)
        time_offset = timedelta(
            hours=np.random.uniform(0, 6.5),
            days=np.random.randint(0, 30)
        )
        entity = f"ENT-{np.random.randint(1, n_entities + 1):04d}"
        symbol = np.random.choice(SYMBOLS)

        # realistic base prices per symbol
        base_prices = {
            "AAPL": 182, "MSFT": 405, "TSLA": 245, "GOOG": 141,
            "AMZN": 178, "META": 390, "NVDA": 680, "JPM": 195
        }
        price = base_prices[symbol] * np.random.uniform(0.97, 1.03)

        records.append({
            "timestamp": base_time + time_offset,
            "entity_id": entity,
            "symbol": symbol,
            "trade_type": np.random.choice(["buy", "sell"]),
            "price": round(price, 2),
            "quantity": int(np.random.lognormal(mean=4, sigma=1)),
            "is_cancelled": np.random.random() < 0.02,  # 2% normal cancel rate
            "counterparty": f"ENT-{np.random.randint(1, n_entities + 1):04d}",
            "is_fraud": False,
            "fraud_type": None,
        })

    return pd.DataFrame(records)


def _generate_fraud_trades(n_fraud, seed):
    """Inject fraud trades across 5 pattern types."""
    np.random.seed(seed + 100)

    fraud_types = config.FRAUD_TYPES
    per_type = n_fraud // len(fraud_types)

    all_fraud = []

    for ftype in fraud_types:
        if ftype == "wash_trading":
            all_fraud.extend(_inject_wash_trading(per_type, seed))
        elif ftype == "spoofing":
            all_fraud.extend(_inject_spoofing(per_type, seed))
        elif ftype == "pump_and_dump":
            all_fraud.extend(_inject_pump_and_dump(per_type, seed))
        elif ftype == "insider_trading":
            all_fraud.extend(_inject_insider_trading(per_type, seed))
        elif ftype == "layering":
            all_fraud.extend(_inject_layering(per_type, seed))

    return pd.DataFrame(all_fraud)


def _inject_wash_trading(n, seed):
    """Same entity on both sides, high volume, minimal price change."""
    records = []
    base_time = datetime(2024, 1, 15, 9, 30, 0)

    for i in range(n):
        entity = f"ENT-{np.random.randint(1, 50):04d}"
        symbol = np.random.choice(SYMBOLS[:4])
        base_price = {"AAPL": 182, "MSFT": 405, "TSLA": 245, "GOOG": 141}[symbol]
        time_offset = timedelta(hours=np.random.uniform(0, 6.5), days=np.random.randint(0, 30))

        records.append({
            "timestamp": base_time + time_offset,
            "entity_id": entity,
            "symbol": symbol,
            "trade_type": np.random.choice(["buy", "sell"]),
            "price": round(base_price * np.random.uniform(0.999, 1.001), 2),  # almost no price change
            "quantity": int(np.random.lognormal(mean=7, sigma=0.5)),  # unusually large
            "is_cancelled": False,
            "counterparty": entity,  # SELF-TRADE — key signal
            "is_fraud": True,
            "fraud_type": "wash_trading",
        })
    return records


def _inject_spoofing(n, seed):
    """Large orders that get cancelled rapidly."""
    records = []
    base_time = datetime(2024, 1, 15, 9, 30, 0)

    for i in range(n):
        entity = f"ENT-{np.random.randint(1, 50):04d}"
        symbol = np.random.choice(SYMBOLS[4:])
        base_price = {"AMZN": 178, "META": 390, "NVDA": 680, "JPM": 195}[symbol]
        time_offset = timedelta(hours=np.random.uniform(0, 6.5), days=np.random.randint(0, 30))

        records.append({
            "timestamp": base_time + time_offset,
            "entity_id": entity,
            "symbol": symbol,
            "trade_type": np.random.choice(["buy", "sell"]),
            "price": round(base_price * np.random.uniform(0.98, 1.02), 2),
            "quantity": int(np.random.lognormal(mean=8, sigma=0.3)),  # very large
            "is_cancelled": True,  # ALWAYS CANCELLED — key signal
            "counterparty": f"ENT-{np.random.randint(51, 200):04d}",
            "is_fraud": True,
            "fraud_type": "spoofing",
        })
    return records


# def _inject_pump_and_dump(n, seed):
#     """Aggressive buying pushing price up, then large sell-off."""
#     records = []
#     base_time = datetime(2024, 1, 15, 9, 30, 0)

#     # generate in clusters of 4 (3 buys + 1 dump)
#     for i in range(n):
#         entity = f"ENT-{np.random.randint(1, 30):04d}"
#         symbol = np.random.choice(SYMBOLS[:3])
#         base_price = {"AAPL": 182, "MSFT": 405, "TSLA": 245}[symbol]
#         day = np.random.randint(0, 30)
#         hour = np.random.uniform(1, 5)

#         # price escalation
#         price_bump = 1 + (i % 4) * np.random.uniform(0.02, 0.05)
#         is_dump = (i % 4 == 3)

#         records.append({
#             "timestamp": base_time + timedelta(hours=hour + (i % 4) * 0.1, days=day),
#             "entity_id": entity,
#             "symbol": symbol,
#             "trade_type": "sell" if is_dump else "buy",
#             "price": round(base_price * price_bump, 2),
#             "quantity": int(np.random.lognormal(mean=7, sigma=0.5)),
#             "is_cancelled": False,
#             "counterparty": f"ENT-{np.random.randint(51, 200):04d}",
#             "is_fraud": True,
#             "fraud_type": "pump_and_dump",
#         })
#     return records

def _inject_pump_and_dump(n_clusters=50, seed=None):
    """
    Generate synthetic pump-and-dump trade records.

    Each cluster contains:
        - 3 aggressive buys (price increasing)
        - 1 large sell (dump)
    """

    if seed is not None:
        np.random.seed(seed)

    records = []
    base_time = datetime(2024, 1, 15, 9, 30, 0)

    for _ in range(n_clusters):

        # --- consistent actor & symbol per cluster ---
        entity = f"ENT-{np.random.randint(1, 30):04d}"
        symbol = np.random.choice(SYMBOLS[:3])

        base_price = {
            "AAPL": 182,
            "MSFT": 405,
            "TSLA": 245
        }[symbol]

        # --- cluster time base ---
        day_offset = np.random.randint(0, 30)
        hour_offset = np.random.uniform(0.5, 5)

        # --- escalating price increments ---
        price_increments = np.cumsum(
            np.random.uniform(0.02, 0.05, size=3)
        )

        # --- BUY PHASE (Pump) ---
        for step in range(3):

            timestamp = base_time + timedelta(
                days=day_offset,
                hours=hour_offset + step * 0.05  # tightly clustered
            )

            price = round(base_price * (1 + price_increments[step]), 2)

            records.append({
                "timestamp": timestamp,
                "entity_id": entity,
                "symbol": symbol,
                "trade_type": "buy",
                "price": price,
                "quantity": int(np.random.lognormal(mean=7, sigma=0.5)),
                "is_cancelled": False,
                "counterparty": f"ENT-{np.random.randint(100, 300):04d}",
                "is_fraud": True,
                "fraud_type": "pump_and_dump",
            })

        # --- DUMP PHASE (Large Sell-Off) ---
        dump_timestamp = base_time + timedelta(
            days=day_offset,
            hours=hour_offset + 0.2
        )

        dump_price = round(base_price * (1 + price_increments[-1]), 2)

        records.append({
            "timestamp": dump_timestamp,
            "entity_id": entity,
            "symbol": symbol,
            "trade_type": "sell",
            "price": dump_price,
            "quantity": int(np.random.lognormal(mean=8, sigma=0.6)),  # larger sell
            "is_cancelled": False,
            "counterparty": f"ENT-{np.random.randint(300, 600):04d}",
            "is_fraud": True,
            "fraud_type": "pump_and_dump",
        })

    return records


def _inject_insider_trading(n, seed):
    """Sudden burst of activity from a normally quiet entity."""
    records = []
    base_time = datetime(2024, 1, 15, 9, 30, 0)

    for i in range(n):
        # use a small set of entities to create clustering
        entity = f"ENT-{np.random.randint(1, 20):04d}"
        symbol = "NVDA"
        base_price = 720

        # cluster trades in last hour of trading day (suspicious timing)
        time_offset = timedelta(
            hours=np.random.uniform(5.5, 6.5),  # end of day
            days=np.random.choice([10, 11, 12])  # cluster on specific days
        )

        records.append({
            "timestamp": base_time + time_offset,
            "entity_id": entity,
            "symbol": symbol,
            "trade_type": "buy",  # insiders typically buy before good news
            "price": round(base_price * np.random.uniform(0.99, 1.01), 2),
            "quantity": int(np.random.lognormal(mean=7, sigma=0.8)),  # large
            "is_cancelled": False,
            "counterparty": f"ENT-{np.random.randint(51, 200):04d}",
            "is_fraud": True,
            "fraud_type": "insider_trading",
        })
    return records


def _inject_layering(n, seed):
    """Multiple orders at different price levels, then cancelled."""
    records = []
    base_time = datetime(2024, 1, 15, 9, 30, 0)

    for i in range(n):
        entity = f"ENT-{np.random.randint(1, 40):04d}"
        symbol = np.random.choice(SYMBOLS)
        base_price = {
            "AAPL": 182, "MSFT": 405, "TSLA": 245, "GOOG": 141,
            "AMZN": 178, "META": 390, "NVDA": 680, "JPM": 195
        }[symbol]
        day = np.random.randint(0, 30)

        # price at different levels — stacked
        level = (i % 5) + 1
        price = base_price + level * 0.50

        records.append({
            "timestamp": base_time + timedelta(
                hours=np.random.uniform(0, 6.5),
                days=day,
                seconds=level  # rapid succession
            ),
            "entity_id": entity,
            "symbol": symbol,
            "trade_type": "sell",
            "price": round(price, 2),
            "quantity": int(level * np.random.lognormal(mean=5, sigma=0.5)),  # increasing size
            "is_cancelled": True,  # all cancelled
            "counterparty": f"ENT-{np.random.randint(51, 200):04d}",
            "is_fraud": True,
            "fraud_type": "layering",
        })
    return records


# ── External Data Loading ──

def load_external_data(filepath):
    """
    Load trading data from CSV or JSON file.
    Validates schema and normalizes to unified format.
    
    Args:
        filepath: path to CSV or JSON file
        
    Returns:
        pd.DataFrame with unified schema
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use CSV or JSON.")

    # ── Validate required columns ──
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Fill optional columns with defaults ──
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            if col == "trade_id":
                df[col] = [f"TXN-{i:06d}" for i in range(len(df))]
            elif col == "amount":
                df[col] = df["price"] * df["quantity"]
            else:
                df[col] = default

    # ── Normalize types ──
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["price"] = df["price"].astype(float)
    df["quantity"] = df["quantity"].astype(int)
    df["is_cancelled"] = df["is_cancelled"].astype(bool)
    df["is_fraud"] = df["is_fraud"].astype(bool)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def get_data(source="synthetic", filepath=None, n_rows=None):
    """
    Unified entry point — get data from any source.
    
    Args:
        source: "synthetic" or "file"
        filepath: path to external file (required if source="file")
        n_rows: number of rows for synthetic data
        
    Returns:
        pd.DataFrame with unified schema
    """
    if source == "synthetic":
        df = generate_synthetic_data(n_rows=n_rows)
        print(f"Generated {len(df)} synthetic trades")
        print(f"  Normal: {len(df[~df['is_fraud']])}")
        print(f"  Fraud:  {len(df[df['is_fraud']])}")
        fraud_counts = df[df["is_fraud"]]["fraud_type"].value_counts()
        for ftype, count in fraud_counts.items():
            print(f"    {ftype}: {count}")
        return df

    elif source == "file":
        if filepath is None:
            raise ValueError("filepath required when source='file'")
        df = load_external_data(filepath)
        print(f"Loaded {len(df)} trades from {filepath}")
        return df

    else:
        raise ValueError(f"Unknown source: {source}. Use 'synthetic' or 'file'.")


# ── Quick test ──
if __name__ == "__main__":
    df = get_data(source="synthetic", n_rows=10000)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample normal trade:")
    print(df[~df["is_fraud"]].iloc[0].to_dict())
    print(f"\nSample fraud trade:")
    print(df[df["is_fraud"]].iloc[0].to_dict())
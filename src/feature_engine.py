"""
Feature Engineering Module
Extracts 12 behavioral features from raw trading data at 3 levels:
  - Transaction-level: individual trade characteristics
  - Entity-level: trader behavior patterns
  - Market-level: symbol/market-wide signals

Each feature is designed to surface specific fraud signals.
"""

import pandas as pd
import numpy as np


#Feature columns used by downstream detectors and scorers
FEATURE_COLUMNS = [
    "amount_zscore",
    "quantity_zscore",
    "is_self_trade",
    "hour_of_day",
    "time_since_last_trade",
    "entity_trade_frequency",
    "entity_cancel_rate",
    "quantity_vs_entity_avg",
    "price_vs_symbol_avg",
    "price_velocity",
    "symbol_volume_zscore",
    "symbol_volatility",
]

# Human-readable descriptions for explainability
FEATURE_DESCRIPTIONS = {
    "amount_zscore": "Transaction amount deviation from average",
    "quantity_zscore": "Trade quantity deviation from average",
    "is_self_trade": "Both sides of trade are same entity",
    "hour_of_day": "Hour when trade was executed",
    "time_since_last_trade": "Seconds since entity's previous trade",
    "entity_trade_frequency": "Entity's average trades per day",
    "entity_cancel_rate": "Fraction of entity's cancelled orders",
    "quantity_vs_entity_avg": "Trade size vs entity's typical size",
    "price_vs_symbol_avg": "Price deviation from symbol average",
    "price_velocity": "Rate of price change per second",
    "symbol_volume_zscore": "Daily volume deviation for symbol",
    "symbol_volatility": "Intraday price volatility for symbol",
}


def engineer_features(df):
    """
    Main entry point. Adds 12 features to the trading DataFrame.

    Args:
        df: DataFrame from data_ingestion module (unified schema)

    Returns:
        DataFrame with all original columns + 12 feature columns
    """
    df = df.copy()
    # df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = _add_transaction_features(df)
    df = _add_entity_features(df)
    df = _add_market_features(df)
    df = _fill_missing(df)

    # Final sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_feature_matrix(df):
    """
    Extract only the 12 feature columns as a clean DataFrame.
    Used by detectors as model input.

    Args:
        df: DataFrame that has already been through engineer_features()

    Returns:
        DataFrame with only FEATURE_COLUMNS
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Run engineer_features() first.")
    return df[FEATURE_COLUMNS].copy()


# Helpers
def _safe_zscore(series):
    """Compute z-score with safe handling for zero std and NaN."""
    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0 or pd.isna(std_val) or pd.isna(mean_val):
        return pd.Series(0.0, index=series.index)
    return (series - mean_val) / std_val


def _safe_divide(numerator, denominator, fill_value=0.0):
    """Safe division that handles zero denominators."""
    denom_safe = denominator.replace(0, np.nan)
    result = numerator / denom_safe
    return result.fillna(fill_value)


# Transaction-Level Features 

def _add_transaction_features(df):
    """
    Features computed directly from each individual transaction.
    No grouping needed — pure row-level calculations.
    """
    # How unusual is this transaction's dollar amount?
    df["amount_zscore"] = _safe_zscore(df["amount"])

    # How unusual is this trade's quantity?
    df["quantity_zscore"] = _safe_zscore(df["quantity"])

    # Is the trader trading with themselves? (wash trading signal)
    df["is_self_trade"] = (df["entity_id"] == df["counterparty"]).astype(int)

    # Trading hour (unusual timing signal)
    df["hour_of_day"] = df["timestamp"].dt.hour

    return df


#  Entity-Level Features 

def _add_entity_features(df):
    """
    Features based on each trader's behavioral patterns.
    Compares each trade against the entity's historical norms.
    """
    # Sort by entity then time for sequential calculations
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # Seconds since this entity's previous trade (burst detection)
    df["time_since_last_trade"] = (
        df.groupby("entity_id")["timestamp"]
        .diff()
        .dt.total_seconds() 
    )

    # Compute entity-level aggreate statistics
    entity_stats = df.groupby("entity_id").agg(
        _count=("trade_id", "count"),
        _mean_qty=("quantity", "mean"),
        _cancel_rate=("is_cancelled", "mean"),
        _first_ts=("timestamp", "min"),
        _last_ts=("timestamp", "max"),
    ).reset_index()

    # Trades per day (high frequency = suspicious)
    active_days = (
        (entity_stats["_last_ts"] - entity_stats["_first_ts"])
        .dt.days
        .clip(lower=1)
    )
    entity_stats["entity_trade_frequency"] = entity_stats["_count"] / active_days

    # Merge entity stats back to each trade
    df = df.merge(
        entity_stats[["entity_id", "entity_trade_frequency", "_cancel_rate", "_mean_qty"]],
        on="entity_id",
        how="left",
    )

    # Entity cancel rate (high = spoofing/layering signal)
    df["entity_cancel_rate"] = df["_cancel_rate"]

    # Trade size relative to entity's average (high = unusual large order)
    df["quantity_vs_entity_avg"] = _safe_divide(
        df["quantity"].astype(float),
        df["_mean_qty"]
    )

    # Drop temporary columns
    df = df.drop(columns=["_cancel_rate", "_mean_qty"])

    return df


# Market-Level Features 

def _add_market_features(df):
    """
    Features based on symbol/market-wide patterns.
    Detects unusual price movement, volume spikes, and volatility.
    """
    df["_trade_date"] = df["timestamp"].dt.date

    # Price deviation from symbol's normal range 
    sym_price = df.groupby("symbol")["price"].agg(["mean", "std"]).reset_index()
    sym_price.columns = ["symbol", "_sp_mean", "_sp_std"]
    sym_price["_sp_std"] = sym_price["_sp_std"].fillna(1.0).replace(0, 1.0)

    df = df.merge(sym_price, on="symbol", how="left")
    df["price_vs_symbol_avg"] = (df["price"] - df["_sp_mean"]) / df["_sp_std"]

    # Daily volume and volatility per symbol 
    daily = df.groupby(["symbol", "_trade_date"]).agg(
        _d_vol=("quantity", "sum"),
        _d_pstd=("price", "std"),
    ).reset_index()
    daily["_d_pstd"] = daily["_d_pstd"].fillna(0.0)

    # Volume z-score: is today's volume unusual for this   symbol?
    sym_vol = daily.groupby("symbol")["_d_vol"].agg(["mean", "std"]).reset_index()
    sym_vol.columns = ["symbol", "_sv_mean", "_sv_std"]
    sym_vol["_sv_std"] = sym_vol["_sv_std"].fillna(1.0).replace(0, 1.0)

    daily = daily.merge(sym_vol, on="symbol", how="left")
    daily["symbol_volume_zscore"] = (daily["_d_vol"] - daily["_sv_mean"]) / daily["_sv_std"]
    daily["symbol_volatility"] = daily["_d_pstd"]

    # Merge daily stats back (each trade gets its day's stats)
    df = df.merge(
        daily[["symbol", "_trade_date", "symbol_volume_zscore", "symbol_volatility"]],
        on=["symbol", "_trade_date"],
        how="left",
    )

    # Price velocity 
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    time_delta = df.groupby("symbol")["timestamp"].diff().dt.total_seconds()
    price_delta = df.groupby("symbol")["price"].diff()
    df["price_velocity"] = _safe_divide(price_delta, time_delta)

    # Drop ALL temporary columns 
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=temp_cols)

    return df


def _fill_missing(df):
    """Fill NaN values in all feature columns with 0."""
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


# Self-Test 

if __name__ == "__main__":
    from data_ingestion import get_data

    print("=" * 60)
    print("  FEATURE ENGINEERING — SELF TEST")
    print("=" * 60)

    # Generate test data
    df = get_data(source="synthetic", n_rows=1000)
    print(f"\nRaw data shape: {df.shape}")
    print(f"Raw columns: {list(df.columns)}")

    # Engineer features
    df = engineer_features(df)
    print(f"\nAfter feature engineering: {df.shape}")

    # Verify all feature columns exist and have no nulls
    print(f"\n{'Feature':<32} {'Min':>10} {'Max':>10} {'Nulls':>6}")
    print("-" * 62)
    all_ok = True
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            print(f"  MISSING: {col}")
            all_ok = False
            continue
        nulls = df[col].isna().sum()
        print(f"  {col:<30} {df[col].min():10.3f} {df[col].max():10.3f} {nulls:6d}")
        if nulls > 0:
            all_ok = False

    # Test feature matrix extraction
    feat_matrix = get_feature_matrix(df)
    print(f"\nFeature matrix shape: {feat_matrix.shape}")

    # Compare fraud vs normal — the whole point of features
    print(f"\n{'Feature':<32} {'Normal':>10} {'Fraud':>10} {'Diff':>10} {'Signal'}")
    print("-" * 78)
    fraud_mask = df["is_fraud"]
    for col in FEATURE_COLUMNS:
        n_mean = df.loc[~fraud_mask, col].mean()
        f_mean = df.loc[fraud_mask, col].mean()
        diff = f_mean - n_mean
        signal = " <<<" if abs(diff) > 0.3 else ""
        print(f"  {col:<30} {n_mean:10.3f} {f_mean:10.3f} {diff:+10.3f}{signal}")

    print(f"\n{'PASS' if all_ok else 'FAIL'}: All features generated with zero nulls")
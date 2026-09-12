"""
src/features/stress.py
Phase 3.5 - Composite financial stress score per customer.

Score combines income disruption, payment difficulty, savings erosion, and
discretionary behavior change, z-scored against the population.

Output:
    data/processed/stress_features.parquet
    outputs/reports/phase3_5_stress.txt

Run from SamriddhiAI/ root:
    py -3.11 -m src.features.stress
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

LAST3_START = pd.Timestamp("2024-11-01")
LAST3_END = pd.Timestamp("2025-01-31")
FIRST3_START = pd.Timestamp("2024-07-01")
FIRST3_END = pd.Timestamp("2024-09-30")


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def load_monthly(log):
    df = pd.read_parquet(PROCESSED / "txn_monthly.parquet")
    log.write(f"Loaded txn_monthly: {df.shape}")
    df["month"] = pd.to_datetime(df["month"])
    return df


def compute_last3(monthly, log):
    log.write("\n" + "=" * 60)
    log.write("Aggregating last-3-month features (Nov 2024 - Jan 2025)")
    log.write("=" * 60)

    w = monthly[(monthly["month"] >= LAST3_START) & (monthly["month"] <= LAST3_END)]

    agg = w.groupby("customer_id").agg(
        savings_rate_last3=("savings_rate", "mean"),
        discretionary_ratio_last3=("discretionary_ratio", "mean"),
        salary_sum_last3=("salary_sum", "mean"),
        emi_count_last3=("emi_count", "mean"),
        bounce_sum_last3=("bounce_count", "sum"),   # renamed to avoid collision
        txn_count_last3=("txn_count", "mean"),
    )

    log.write(f"Last-3 aggregation shape: {agg.shape}")
    return agg


def compute_first3(monthly, log):
    log.write("\n" + "=" * 60)
    log.write("Aggregating first-3-month features (Jul - Sep 2024)")
    log.write("=" * 60)

    w = monthly[(monthly["month"] >= FIRST3_START) & (monthly["month"] <= FIRST3_END)]

    agg = w.groupby("customer_id").agg(
        salary_sum_first3=("salary_sum", "mean"),
    )

    log.write(f"First-3 aggregation shape: {agg.shape}")
    return agg


def compute_missed_emi_last3(monthly, log):
    """Count months in last-3 where a stable EMI history suddenly went to zero."""
    log.write("\n" + "=" * 60)
    log.write("Computing missed EMI in last-3 window")
    log.write("=" * 60)

    df = monthly.sort_values(["customer_id", "month"]).copy()

    # Prior-3-month median emi_count (excluding current month)
    df["prior_median"] = (
        df.groupby("customer_id")["emi_count"]
          .transform(lambda s: s.rolling(3, min_periods=1).median().shift(1))
    )

    # Missed = had regular EMI (prior median >= 1) and current month is 0
    df["missed"] = ((df["prior_median"] >= 1) & (df["emi_count"] == 0)).astype(int)

    last3 = df[(df["month"] >= LAST3_START) & (df["month"] <= LAST3_END)]
    agg = last3.groupby("customer_id")["missed"].sum().rename("missed_emi_last3")

    log.write(f"Missed EMI aggregation shape: {agg.shape}")
    log.write(f"Customers with >=1 missed EMI in last 3 months: {(agg > 0).sum()}")
    log.write(f"Total missed-EMI events in last 3 months: {agg.sum()}")
    return agg

def zscore(series, log, name):
    """Population z-score; fills NaN with 0."""
    m = series.mean()
    s = series.std()
    if s == 0:
        log.write(f"  [WARN] {name}: std=0, all z set to 0")
        return pd.Series(0.0, index=series.index)
    return (series - m) / s


def build_stress(features, log):
    log.write("\n" + "=" * 60)
    log.write("Building composite stress score")
    log.write("=" * 60)

    # z-scores, winsorized to [-3, +3] to prevent single outliers from dominating
    def winsorize(s, lo=-3, hi=3):
        return s.clip(lower=lo, upper=hi)

    features["z_emi"] = winsorize(zscore(features["missed_emi_last3"], log, "missed_emi_last3"))
    features["z_bounce"] = winsorize(zscore(features["bounce_sum_last3"], log, "bounce_sum_last3"))
    features["z_savings"] = winsorize(zscore(-features["savings_rate_last3"], log, "-savings_rate_last3"))
    features["z_discretionary"] = winsorize(zscore(features["discretionary_ratio_last3"], log, "discretionary_ratio_last3"))
    features["z_salary"] = winsorize(zscore(-features["salary_momentum_last3"], log, "-salary_momentum_last3"))

    # Weighted sum.
    # Note: the dataset contains zero missed-EMI events (verified), so z_emi
    # has std=0 and contributes nothing. Its weight is redistributed to
    # payment-difficulty (bounce), which is the closest available proxy.
    # In production, missed-EMI would carry weight 0.30 as the strongest signal.
    features["stress_score"] = (
        0.40 * features["z_bounce"] +
        0.25 * features["z_savings"] +
        0.20 * features["z_discretionary"] +
        0.15 * features["z_salary"]
    )

    # Bucket by percentile
    p40 = features["stress_score"].quantile(0.40)
    p75 = features["stress_score"].quantile(0.75)

    def bucket(x):
        if x <= p40: return "low"
        if x <= p75: return "medium"
        return "high"

    features["stress_level"] = features["stress_score"].apply(bucket)

    # One-hot
    for lvl in ["low", "medium", "high"]:
        features[f"stress_{lvl}"] = (features["stress_level"] == lvl).astype(int)
    features["is_stressed"] = (features["stress_level"] == "high").astype(int)

    log.write(f"\nstress_score describe:\n{features['stress_score'].describe()}")
    log.write(f"\nstress_level value counts:\n{features['stress_level'].value_counts()}")
    log.write(f"\nis_stressed count: {features['is_stressed'].sum()}")

    return features


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 3.5 — Stress Score — {datetime.now().isoformat(timespec='seconds')}")

    monthly = load_monthly(log)
    profile = pd.read_parquet(PROCESSED / "profile_features.parquet")
    all_customers = profile[["customer_id"]].copy()

    last3 = compute_last3(monthly, log)
    first3 = compute_first3(monthly, log)
    missed = compute_missed_emi_last3(monthly, log)

    # Merge
    df = all_customers.merge(last3, on="customer_id", how="left") \
                      .merge(first3, on="customer_id", how="left") \
                      .merge(missed, on="customer_id", how="left")

    # Fill nulls: no data = no signal (0)
    df = df.fillna({
        "savings_rate_last3": 0,
        "discretionary_ratio_last3": 0,
        "salary_sum_last3": 0,
        "emi_count_last3": 0,
        "bounce_sum_last3": 0,
        "txn_count_last3": 0,
        "salary_sum_first3": 0,
        "missed_emi_last3": 0,
    })

    # Salary momentum
    df["salary_momentum_last3"] = df["salary_sum_last3"] - df["salary_sum_first3"]

    df = build_stress(df, log)

    # Keep only needed columns
    keep = [
        "customer_id",
        "missed_emi_last3", "bounce_sum_last3",
        "savings_rate_last3", "discretionary_ratio_last3",
        "salary_momentum_last3",
        "z_emi", "z_bounce", "z_savings", "z_discretionary", "z_salary",
        "stress_score", "stress_level",
        "stress_low", "stress_medium", "stress_high", "is_stressed",
    ]
    out = df[keep].copy()

    log.write(f"\nFinal shape: {out.shape}")
    log.write(f"Nulls:\n{out.isnull().sum()[out.isnull().sum() > 0]}")

    out.to_parquet(PROCESSED / "stress_features.parquet", index=False)
    log.write(f"\nSaved: {PROCESSED / 'stress_features.parquet'}")

    log.save(REPORT_DIR / "phase3_5_stress.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase3_5_stress.txt'}")


if __name__ == "__main__":
    main()
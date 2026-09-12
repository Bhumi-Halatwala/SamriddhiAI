"""
src/features/txn_features.py
Phase 3.2 - Build transaction features.

Outputs:
    data/processed/txn_monthly.parquet    (customer_id, month, ~45 features)
    data/processed/txn_snapshot.parquet   (customer_id, ~60 features, as of Jan 2025)
    outputs/reports/phase3_2_txn.txt

Run from SamriddhiAI/ root:
    py -3.11 -m src.features.txn_features
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

CATEGORIES = ["salary", "business_income", "emi", "rent",
              "discretionary", "bounce", "suspicious"]

CHANNELS = ["UPI", "NEFT", "ECS"]

SNAPSHOT_CUTOFF = pd.Timestamp("2025-01-31")
SNAPSHOT_START = pd.Timestamp("2024-07-01")
FULL_START = pd.Timestamp("2024-01-01")
FULL_END = pd.Timestamp("2025-05-31")   # exclude June (partial)


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def load_ledger(log):
    df = pd.read_parquet(INTERIM / "transaction_ledger.parquet")
    log.write(f"Loaded ledger: {df.shape}")
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"])
    return df


# ---------------------------------------------------------------
# Monthly aggregation
# ---------------------------------------------------------------
def monthly_aggregate(ledger, log):
    log.write("\n" + "=" * 60)
    log.write("Building monthly aggregates")
    log.write("=" * 60)

    # Filter to months we trust (drop June 2025 partial)
    ledger = ledger[
        (ledger["date"] >= FULL_START) & (ledger["date"] <= FULL_END)
    ].copy()

    ledger["month"] = ledger["date"].dt.to_period("M").dt.to_timestamp()
    ledger["is_credit"] = (ledger["direction"] == "credit").astype(int)
    ledger["abs_amount"] = ledger["amount"].abs()

    log.write(f"Filtered ledger rows: {len(ledger):,}")
    log.write(f"Months covered: {ledger['month'].nunique()}")
    log.write(f"Month range: {ledger['month'].min()} to {ledger['month'].max()}")

    # ---- base volume per (customer, month) ----
    base = ledger.groupby(["customer_id", "month"]).agg(
        txn_count=("amount", "size"),
        credit_count=("is_credit", "sum"),
        credit_sum=("amount", lambda s: s[s > 0].sum()),
        debit_sum=("amount", lambda s: -s[s < 0].sum()),
        max_credit=("amount", lambda s: s[s > 0].max() if (s > 0).any() else 0.0),
        max_debit=("amount", lambda s: (-s[s < 0].min()) if (s < 0).any() else 0.0),
    ).reset_index()

    base["debit_count"] = base["txn_count"] - base["credit_count"]
    base["net_flow"] = base["credit_sum"] - base["debit_sum"]
    base["avg_credit"] = base["credit_sum"] / base["credit_count"].replace(0, np.nan)
    base["avg_debit"] = base["debit_sum"] / base["debit_count"].replace(0, np.nan)
    base["avg_credit"] = base["avg_credit"].fillna(0.0)
    base["avg_debit"] = base["avg_debit"].fillna(0.0)

    # ---- category counts and sums ----
    for cat in CATEGORIES:
        cat_df = ledger[ledger["true_category"] == cat]
        agg = cat_df.groupby(["customer_id", "month"]).agg(
            **{
                f"{cat}_count": ("amount", "size"),
                f"{cat}_sum": ("abs_amount", "sum"),
            }
        ).reset_index()
        base = base.merge(agg, on=["customer_id", "month"], how="left")

    # ---- channel counts ----
    for ch in CHANNELS:
        ch_df = ledger[ledger["channel"] == ch]
        agg = ch_df.groupby(["customer_id", "month"]).size().reset_index(name=f"{ch.lower()}_count")
        base = base.merge(agg, on=["customer_id", "month"], how="left")

    # ---- fill nulls from left joins ----
    for cat in CATEGORIES:
        base[f"{cat}_count"] = base[f"{cat}_count"].fillna(0).astype(int)
        base[f"{cat}_sum"] = base[f"{cat}_sum"].fillna(0.0)
    for ch in CHANNELS:
        base[f"{ch.lower()}_count"] = base[f"{ch.lower()}_count"].fillna(0).astype(int)

    # ---- derived ratios ----
    base["total_flow"] = base["credit_sum"] + base["debit_sum"]
    base["savings_rate"] = base["net_flow"] / (base["credit_sum"] + 1)
    base["salary_share"] = base["salary_sum"] / (base["credit_sum"] + 1)
    base["discretionary_ratio"] = base["discretionary_sum"] / (base["debit_sum"] + 1)
    base["bounce_rate"] = base["bounce_count"] / base["txn_count"].clip(lower=1)
    base["upi_share"] = base["upi_count"] / base["txn_count"].clip(lower=1)
    base["neft_share"] = base["neft_count"] / base["txn_count"].clip(lower=1)
    base["ecs_share"] = base["ecs_count"] / base["txn_count"].clip(lower=1)

    # Category shares (relative to total flow)
    for cat in CATEGORIES:
        base[f"{cat}_share"] = base[f"{cat}_sum"] / (base["total_flow"] + 1)

    log.write(f"Monthly aggregates shape: {base.shape}")
    log.write(f"Columns: {list(base.columns)}")

    return base


# ---------------------------------------------------------------
# Add income-relative ratios (needs monthly income)
# ---------------------------------------------------------------
def add_income_ratios(monthly, profile, log):
    log.write("\nAdding income-relative ratios...")

    income = profile[["customer_id", "monthly_declared_income"]].copy()
    monthly = monthly.merge(income, on="customer_id", how="left")

    monthly["emi_to_income"] = monthly["emi_sum"] / (monthly["monthly_declared_income"] + 1)
    monthly["rent_to_income"] = monthly["rent_sum"] / (monthly["monthly_declared_income"] + 1)

    return monthly


# ---------------------------------------------------------------
# Snapshot aggregation (Jan 2025)
# ---------------------------------------------------------------
def snapshot_from_monthly(monthly, log):
    log.write("\n" + "=" * 60)
    log.write("Building snapshot as of Jan 2025 (7-month window: Jul 2024 - Jan 2025)")
    log.write("=" * 60)

    window = monthly[
        (monthly["month"] >= SNAPSHOT_START) & (monthly["month"] <= SNAPSHOT_CUTOFF)
    ].copy()

    log.write(f"Snapshot window rows: {len(window):,}")
    log.write(f"Expected ~5000 customers × 7 months = 35000; got {len(window)}")

    # ----- Features to aggregate with mean / slope / std / momentum -----
    agg_targets = [
        "txn_count", "credit_sum", "debit_sum", "net_flow",
        "savings_rate", "salary_sum", "emi_sum", "rent_sum",
        "discretionary_sum", "bounce_count", "salary_share",
        "discretionary_ratio", "emi_to_income", "upi_share",
    ]

    # ----- Mean over 7 months -----
    mean_agg = window.groupby("customer_id")[agg_targets].mean() \
                     .add_suffix("_mean_7")

    # ----- Std over 7 months -----
    std_agg = window.groupby("customer_id")[agg_targets].std() \
                    .add_suffix("_std_7").fillna(0)

    # ----- Last 3 months mean (Nov, Dec, Jan) -----
    last3 = window[window["month"] >= pd.Timestamp("2024-11-01")]
    last3_agg = last3.groupby("customer_id")[agg_targets].mean() \
                     .add_suffix("_last3")

    # ----- First 3 months mean (Jul, Aug, Sep) -----
    first3 = window[window["month"] <= pd.Timestamp("2024-09-30")]
    first3_agg = first3.groupby("customer_id")[agg_targets].mean() \
                       .add_suffix("_first3")

    # ----- Momentum = last3 - first3 -----
    momentum = (last3_agg.values - first3_agg.values)
    momentum = pd.DataFrame(
        momentum,
        index=last3_agg.index,
        columns=[c.replace("_last3", "_momentum") for c in last3_agg.columns],
    )

    # ----- Slope over 7 months -----
    slopes = {}
    months_num = window["month"].astype("int64") / 1e18  # convert to numeric
    window_x = window.assign(t_numeric=months_num)

    for col in agg_targets:
        def slope_fn(g):
            if len(g) < 2: return 0.0
            return np.polyfit(g["t_numeric"].values, g[col].values, 1)[0]
        slopes[col] = window_x.groupby("customer_id").apply(slope_fn)

    slope_df = pd.DataFrame(slopes).add_suffix("_slope_7")

    # ----- Regularity features (custom) -----
    reg = window.groupby("customer_id").agg(
        salary_months_present=("salary_sum", lambda s: (s > 0).sum()),
        emi_months_present=("emi_sum", lambda s: (s > 0).sum()),
        bounce_months_present=("bounce_count", lambda s: (s > 0).sum()),
        zero_salary_months=("salary_sum", lambda s: (s == 0).sum()),
        min_savings_rate=("savings_rate", "min"),
        max_savings_rate=("savings_rate", "max"),
        min_salary_sum=("salary_sum", "min"),
        max_salary_sum=("salary_sum", "max"),
    )

    # Salary regularity: std / mean (lower = more regular)
    reg["salary_cv"] = (
        window.groupby("customer_id")["salary_sum"].std() /
        (window.groupby("customer_id")["salary_sum"].mean() + 1)
    ).fillna(0)

    # ----- Missed EMI within window -----
    def count_missed_emi(g):
        g = g.sort_values("month")
        cnt = 0
        for i in range(3, len(g)):
            prior = g["emi_count"].iloc[i-3:i].median()
            if prior > 0 and g["emi_count"].iloc[i] < prior - 1:
                cnt += 1
        return cnt
    missed = window.groupby("customer_id").apply(count_missed_emi)
    reg["missed_emi_in_window"] = missed

    # ----- Suspicious / window_dressing flags -----
    has_susp = (window.groupby("customer_id")["suspicious_count"].sum() > 0).astype(int)
    has_bounce = (window.groupby("customer_id")["bounce_count"].sum() > 0).astype(int)
    reg["has_window_dressing"] = has_susp
    reg["has_bounce"] = has_bounce

    # ----- Merge everything -----
    snap = mean_agg \
        .join(std_agg, how="outer") \
        .join(last3_agg, how="outer") \
        .join(first3_agg, how="outer") \
        .join(momentum, how="outer") \
        .join(slope_df, how="outer") \
        .join(reg, how="outer") \
        .reset_index()

    log.write(f"Snapshot shape: {snap.shape}")
    log.write(f"Null counts > 0:\n{snap.isnull().sum()[snap.isnull().sum() > 0]}")

    return snap


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 3.2 — Transaction Features — {datetime.now().isoformat(timespec='seconds')}")

    ledger = load_ledger(log)
    profile = pd.read_parquet(PROCESSED / "profile_features.parquet")

    monthly = monthly_aggregate(ledger, log)
    monthly = add_income_ratios(monthly, profile, log)

    # Save monthly
    monthly.to_parquet(PROCESSED / "txn_monthly.parquet", index=False)
    log.write(f"\nSaved: {PROCESSED / 'txn_monthly.parquet'}")

    # Build snapshot
    snapshot = snapshot_from_monthly(monthly, log)
    snapshot.to_parquet(PROCESSED / "txn_snapshot.parquet", index=False)
    log.write(f"\nSaved: {PROCESSED / 'txn_snapshot.parquet'}")

    # Final summary
    log.write("\n" + "=" * 60)
    log.write("SUMMARY")
    log.write("=" * 60)
    log.write(f"Monthly table: {monthly.shape}  ({monthly['customer_id'].nunique()} customers, "
              f"{monthly['month'].nunique()} months)")
    log.write(f"Snapshot table: {snapshot.shape}")
    log.write(f"Snapshot covers all customers: "
              f"{snapshot['customer_id'].nunique() == profile['customer_id'].nunique()}")

    log.save(REPORT_DIR / "phase3_2_txn.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase3_2_txn.txt'}")


if __name__ == "__main__":
    main()
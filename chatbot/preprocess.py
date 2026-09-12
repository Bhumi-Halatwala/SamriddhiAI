"""
Preprocessing: turn the 4 raw files into ONE customer-level feature table.

Run:
    python preprocess.py

Reads from  ./data/  (put the 4 CSVs there)
Writes to   ./data/customer_features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")


def load_raw():
    cm = pd.read_csv(DATA_DIR / "customer_master.csv")
    lb = pd.read_csv(DATA_DIR / "labels.csv")
    bl = pd.read_csv(DATA_DIR / "behavioral_log.csv", parse_dates=["timestamp"])
    tx = pd.read_csv(DATA_DIR / "transaction_ledger.csv", parse_dates=["date"])
    return cm, lb, bl, tx


# ------------------------------------------------------------------
# 1. Transaction ledger -> per-customer financial behavior features
# ------------------------------------------------------------------
def build_transaction_features(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["month"] = tx["date"].dt.to_period("M")

    # total credit/debit by category, per customer
    cat_sums = (
        tx.pivot_table(index="customer_id", columns="true_category", values="amount",
                        aggfunc="sum", fill_value=0)
        .add_prefix("sum_")
    )
    cat_counts = (
        tx.pivot_table(index="customer_id", columns="true_category", values="amount",
                        aggfunc="count", fill_value=0)
        .add_prefix("count_")
    )

    # monthly salary credit (proxy for declared income accuracy / regularity)
    salary = tx[tx.true_category == "salary"]
    salary_monthly = salary.groupby(["customer_id", "month"]).amount.sum().reset_index()
    salary_stats = salary_monthly.groupby("customer_id").amount.agg(
        avg_monthly_salary="mean", salary_months_seen="count", salary_std="std"
    )
    salary_stats["salary_regularity"] = 1 - (
        salary_stats["salary_std"].fillna(0) / salary_stats["avg_monthly_salary"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)

    # business income (self-employed / business-owner customers get paid this way instead of "salary")
    biz = tx[tx.true_category == "business_income"]
    biz_monthly = biz.groupby(["customer_id", "month"]).amount.sum().reset_index()
    biz_stats = biz_monthly.groupby("customer_id").amount.agg(avg_monthly_business_income="mean")

    # EMI burden - compute as a MONTHLY figure (avg of monthly EMI outflow), not a raw multi-year sum,
    # so it can be compared directly against monthly income later
    emi_tx = tx[tx.true_category == "emi"].copy()
    emi_tx["amount"] = emi_tx["amount"].abs()
    emi_monthly = emi_tx.groupby(["customer_id", "month"]).amount.sum().reset_index()
    emi_stats = emi_monthly.groupby("customer_id").amount.agg(avg_monthly_emi="mean")
    total_emi = emi_tx.groupby("customer_id").amount.sum().rename("total_emi_outflow")

    # savings proxy: (credits - discretionary - emi - rent) trend isn't directly available,
    # so use net cashflow per month as a simple proxy
    net_monthly = tx.groupby(["customer_id", "month"]).amount.sum().reset_index()
    net_stats = net_monthly.groupby("customer_id").amount.agg(
        avg_monthly_net="mean", net_volatility="std"
    )

    # recency + count of stress/fraud signal transactions
    bounce = tx[tx.true_category == "bounce"]
    bounce_stats = bounce.groupby("customer_id").agg(
        bounce_count=("amount", "count"),
        last_bounce_date=("date", "max"),
    )
    dataset_end = tx["date"].max()
    if len(bounce_stats):
        bounce_stats["days_since_last_bounce"] = (dataset_end - bounce_stats["last_bounce_date"]).dt.days
        bounce_stats = bounce_stats.drop(columns="last_bounce_date")

    window_dressing_flag = (
        tx[tx.true_category == "window_dressing"].groupby("customer_id").size().rename("window_dressing_count")
    )

    features = (
        cat_sums.join(cat_counts, how="outer")
        .join(salary_stats, how="left")
        .join(biz_stats, how="left")
        .join(emi_stats, how="left")
        .join(total_emi, how="left")
        .join(net_stats, how="left")
        .join(bounce_stats, how="left")
        .join(window_dressing_flag, how="left")
    )

    fill_zero_cols = [
        "bounce_count", "window_dressing_count", "total_emi_outflow", "avg_monthly_emi",
        "salary_months_seen", "avg_monthly_salary", "avg_monthly_business_income",
    ]
    for c in fill_zero_cols:
        if c in features.columns:
            features[c] = features[c].fillna(0)
    features["days_since_last_bounce"] = features.get("days_since_last_bounce", pd.Series(dtype=float)).fillna(9999)
    features["is_window_dressing"] = (features["window_dressing_count"] > 0).astype(int)

    # unified monthly inflow: salary if salaried, else business income - this is the
    # right denominator for EMI-burden ratios regardless of occupation type
    features["avg_monthly_inflow"] = features["avg_monthly_salary"].where(
        features["avg_monthly_salary"] > 0, features["avg_monthly_business_income"]
    )

    return features.reset_index()


# ------------------------------------------------------------------
# 2. Behavioral log -> per-customer engagement / intent features
# ------------------------------------------------------------------
def build_behavioral_features(bl: pd.DataFrame) -> pd.DataFrame:
    event_counts = (
        bl.pivot_table(index="customer_id", columns="event_type", values="event_id",
                        aggfunc="count", fill_value=0)
        .add_prefix("evt_")
    )

    session = bl.groupby("customer_id").session_duration_seconds.agg(
        avg_session_seconds="mean", total_sessions="count"
    )

    # which product they looked at the most -> intent signal
    viewed = bl[bl.product_viewed != "none"]
    top_product = (
        viewed.groupby("customer_id").product_viewed
        .agg(lambda x: x.value_counts().idxmax() if len(x) else "none")
        .rename("most_viewed_product")
    )
    calc_by_product = (
        bl[bl.event_type == "loan_calculator_used"]
        .pivot_table(index="customer_id", columns="product_viewed", values="event_id", aggfunc="count", fill_value=0)
        .add_prefix("calc_used_")
    )

    features = event_counts.join(session, how="left").join(top_product, how="left").join(calc_by_product, how="left")
    features["most_viewed_product"] = features["most_viewed_product"].fillna("none")
    calc_cols = [c for c in features.columns if c.startswith("calc_used_")]
    features[calc_cols] = features[calc_cols].fillna(0)
    return features.reset_index()


# ------------------------------------------------------------------
# 3. Merge everything into one modeling table
# ------------------------------------------------------------------
def build_customer_table():
    cm, lb, bl, tx = load_raw()

    tx_feats = build_transaction_features(tx)
    bl_feats = build_behavioral_features(bl)

    df = (
        cm.merge(tx_feats, on="customer_id", how="left")
        .merge(bl_feats, on="customer_id", how="left")
        .merge(lb, on="customer_id", how="left")
    )

    # engineered ratios that matter for underwriting-style features
    # use REAL observed monthly cashflow (avg_monthly_inflow) as the primary denominator -
    # it reflects actual transaction behavior, not just the self-declared annual figure.
    # Falls back to declared income / 12 only when no salary/business-income transactions exist at all.
    monthly_income_actual = df["avg_monthly_inflow"].where(
        df["avg_monthly_inflow"] > 0, df["declared_annual_income"] / 12
    )
    df["emi_to_income_actual"] = (df["avg_monthly_emi"] / monthly_income_actual).round(3)
    df["emi_to_declared_income"] = (df["avg_monthly_emi"] / (df["declared_annual_income"] / 12)).round(3)
    df["declared_vs_actual_income_gap"] = (
        (df["declared_annual_income"] / 12 - monthly_income_actual) / monthly_income_actual
    ).round(3)
    df["existing_loan_types"] = df["existing_loan_types"].fillna("none")

    return df


if __name__ == "__main__":
    df = build_customer_table()
    out_path = DATA_DIR / "customer_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {out_path}")
    print(df.dtypes)

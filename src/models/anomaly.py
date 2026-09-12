"""
src/models/anomaly.py
Phase 5 - Anomaly & stress detection (v2 — fixed).

Key fix: instead of running Isolation Forest per month on population-normalized
features, we now compute PER-CUSTOMER anomalies:
    - For each customer, z-score each monthly feature against that customer's
      own history.
    - Aggregate into per-customer features: max z, count of z>2 months.
    - Run Isolation Forest on customer-level z-score features.

Also: contamination lowered to 0.03 to match realistic fraud rates.

Run from SamriddhiAI/ root:
    py -3.11 -m src.models.anomaly
"""

from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

sns.set_style("whitegrid")
RANDOM_STATE = 42

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
FIG_DIR = Path("outputs/figures")
REPORT_DIR = Path("outputs/reports")

MONTHLY_FEATURES = [
    "txn_count", "credit_sum", "debit_sum", "net_flow",
    "savings_rate", "salary_sum", "emi_sum", "rent_sum",
    "discretionary_sum", "bounce_count", "suspicious_count",
    "upi_share", "neft_share", "ecs_share",
]


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


# ---------------------------------------------------------------
# Build monthly including June 2025
# ---------------------------------------------------------------
def build_monthly_with_june(log):
    log.write("\n" + "=" * 60)
    log.write("Rebuilding monthly aggregates INCLUDING June 2025")
    log.write("=" * 60)

    ledger = pd.read_parquet(INTERIM / "transaction_ledger.parquet")
    ledger["date"] = pd.to_datetime(ledger["date"])
    ledger["amount"] = pd.to_numeric(ledger["amount"])
    ledger["month"] = ledger["date"].dt.to_period("M").dt.to_timestamp()
    ledger["abs_amount"] = ledger["amount"].abs()

    base = ledger.groupby(["customer_id", "month"]).agg(
        txn_count=("amount", "size"),
        credit_sum=("amount", lambda s: s[s > 0].sum()),
        debit_sum=("amount", lambda s: -s[s < 0].sum()),
    ).reset_index()

    base["net_flow"] = base["credit_sum"] - base["debit_sum"]
    base["savings_rate"] = base["net_flow"] / (base["credit_sum"] + 1)

    for cat in ["salary", "emi", "rent", "discretionary", "bounce", "suspicious"]:
        cdf = ledger[ledger["true_category"] == cat]
        agg = cdf.groupby(["customer_id", "month"]).agg(
            **{
                f"{cat}_sum": ("abs_amount", "sum"),
                f"{cat}_count": ("amount", "size"),
            }
        ).reset_index()
        base = base.merge(agg, on=["customer_id", "month"], how="left")

    for ch in ["UPI", "NEFT", "ECS"]:
        cdf = ledger[ledger["channel"] == ch]
        agg = cdf.groupby(["customer_id", "month"]).size() \
                 .reset_index(name=f"{ch.lower()}_count")
        base = base.merge(agg, on=["customer_id", "month"], how="left")

    for cat in ["salary", "emi", "rent", "discretionary", "bounce", "suspicious"]:
        base[f"{cat}_sum"] = base[f"{cat}_sum"].fillna(0)
        base[f"{cat}_count"] = base[f"{cat}_count"].fillna(0)
    for ch in ["UPI", "NEFT", "ECS"]:
        base[f"{ch.lower()}_count"] = base[f"{ch.lower()}_count"].fillna(0)
        base[f"{ch.lower()}_share"] = base[f"{ch.lower()}_count"] / base["txn_count"].clip(lower=1)

    log.write(f"Rebuilt monthly shape: {base.shape}")
    log.write(f"Months: {base['month'].nunique()}")
    return base


# ---------------------------------------------------------------
# Per-customer anomaly features
# ---------------------------------------------------------------
def per_customer_anomaly_features(monthly, log):
    """
    For each customer, compute per-month z-scores of each feature against
    THAT customer's own history. Then aggregate into anomaly signals.
    """
    log.write("\n" + "=" * 60)
    log.write("Per-customer anomaly feature engineering")
    log.write("=" * 60)

    feats = [c for c in MONTHLY_FEATURES if c in monthly.columns]
    df = monthly.sort_values(["customer_id", "month"]).copy()

    # For each feature, z-score each month vs the customer's own mean/std
    z_feats = []
    for f in feats:
        grp = df.groupby("customer_id")[f]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        col = f"z_{f}"
        df[col] = ((df[f] - mean) / std).fillna(0)
        z_feats.append(col)

    # Count anomalies per customer — a month is anomalous if ANY z > 2.5
    z_cols_abs = df[z_feats].abs()
    df["month_max_z"] = z_cols_abs.max(axis=1)
    df["month_is_anomaly"] = (df["month_max_z"] >= 2.5).astype(int)

    per_cust = df.groupby("customer_id").agg(
        anomaly_months_count=("month_is_anomaly", "sum"),
        max_month_z=("month_max_z", "max"),
        mean_month_z=("month_max_z", "mean"),
        std_month_z=("month_max_z", "std"),
    ).reset_index()

    log.write(f"Per-customer anomaly summary shape: {per_cust.shape}")
    log.write(f"anomaly_months_count describe:\n{per_cust['anomaly_months_count'].describe()}")
    log.write(f"max_month_z describe:\n{per_cust['max_month_z'].describe()}")

    return df, per_cust, z_feats


# ---------------------------------------------------------------
# Isolation Forest on per-customer z-score features
# ---------------------------------------------------------------
def isolation_forest_customer_level(df, per_cust, z_feats, log):
    log.write("\n" + "=" * 60)
    log.write("Isolation Forest on customer-level z-score summaries")
    log.write("=" * 60)

    # For each customer, take the max |z| for each feature across months
    feats_agg = df.groupby("customer_id")[z_feats].apply(
        lambda g: g.abs().max()
    ).reset_index()

    X = feats_agg[z_feats].fillna(0)
    log.write(f"Customer-level feature matrix: {X.shape}")

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.03,   # ~3% matches realistic fraud rate
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X)

    feats_agg["iso_score"] = -iso.decision_function(X)
    feats_agg["iso_flag"] = (iso.predict(X) == -1).astype(int)

    per_cust = per_cust.merge(feats_agg[["customer_id", "iso_score", "iso_flag"]],
                              on="customer_id", how="left")

    log.write(f"Customers flagged by Isolation Forest: {per_cust['iso_flag'].sum()}")
    log.write(f"iso_score describe:\n{per_cust['iso_score'].describe()}")

    return iso, per_cust


# ---------------------------------------------------------------
# Rule-based hard flags
# ---------------------------------------------------------------
def rule_hard_flags(ledger, monthly, stress, log):
    log.write("\n" + "=" * 60)
    log.write("Rule-based hard flags")
    log.write("=" * 60)

    all_cust = stress[["customer_id"]].copy()
    flags = all_cust.copy()

    m = monthly.sort_values(["customer_id", "month"]).copy()
    m["bounce_3m_sum"] = m.groupby("customer_id")["bounce_count"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    bounce_spike = (m.groupby("customer_id")["bounce_3m_sum"].max() >= 2).astype(int)
    flags = flags.merge(
        bounce_spike.rename("hard_bounce").reset_index(),
        on="customer_id", how="left"
    )

    def income_gap(g):
        had_salary = (g["salary_sum"] > 0).sum()
        if had_salary < 4:
            return 0
        is_zero = (g["salary_sum"] == 0).astype(int).values
        max_run, cur = 0, 0
        for v in is_zero:
            cur = cur + 1 if v else 0
            max_run = max(max_run, cur)
        return int(max_run >= 2)

    gap = m.groupby("customer_id", group_keys=False).apply(income_gap).rename(
        "hard_income_gap").reset_index()
    flags = flags.merge(gap, on="customer_id", how="left")

    def savings_drop(g):
        is_neg = (g["savings_rate"] < -0.1).astype(int).values
        max_run, cur = 0, 0
        for v in is_neg:
            cur = cur + 1 if v else 0
            max_run = max(max_run, cur)
        return int(max_run >= 2)

    drop = m.groupby("customer_id", group_keys=False).apply(savings_drop).rename(
        "hard_savings_drop").reset_index()
    flags = flags.merge(drop, on="customer_id", how="left")

    wd = ledger[ledger["true_category"] == "window_dressing"]
    wd_cust = wd["customer_id"].unique()
    flags["hard_wd"] = flags["customer_id"].isin(wd_cust).astype(int)

    flags = flags.merge(
        stress[["customer_id", "is_stressed"]].rename(
            columns={"is_stressed": "hard_stress"}
        ),
        on="customer_id", how="left"
    )

    flag_cols = ["hard_bounce", "hard_income_gap", "hard_savings_drop", "hard_wd", "hard_stress"]
    for c in flag_cols:
        flags[c] = flags[c].fillna(0).astype(int)

    flags["hard_flag_count"] = flags[flag_cols].sum(axis=1)
    flags["is_hard_flagged"] = (flags["hard_flag_count"] >= 1).astype(int)

    for c in flag_cols:
        log.write(f"  {c}: {flags[c].sum()}")
    log.write(f"  is_hard_flagged: {flags['is_hard_flagged'].sum()}")

    return flags


# ---------------------------------------------------------------
# Combine alerts
# ---------------------------------------------------------------
def combine_alerts(per_cust, flags, log):
    log.write("\n" + "=" * 60)
    log.write("Combining detectors")
    log.write("=" * 60)

    df = per_cust.merge(flags, on="customer_id", how="left")

    # Alert level based on multiple signals
    def level(row):
        iso_high = row["iso_flag"] == 1
        many_anom_months = row["anomaly_months_count"] >= 3
        hard = row["hard_flag_count"] >= 1

        if hard and (iso_high or many_anom_months):
            return "high"
        if hard or iso_high or many_anom_months:
            return "high" if hard else "medium"
        return "low"

    df["alert_level"] = df.apply(level, axis=1)
    for lvl in ["low", "medium", "high"]:
        df[f"alert_{lvl}"] = (df["alert_level"] == lvl).astype(int)

    log.write(f"alert_level distribution:\n{df['alert_level'].value_counts()}")
    log.write(f"iso_flag: {df['iso_flag'].sum()}")
    log.write(f"is_hard_flagged: {df['is_hard_flagged'].sum()}")

    return df


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
def eval_ground_truth(df, log):
    log.write("\n" + "=" * 60)
    log.write("Evaluation")
    log.write("=" * 60)

    # window_dressing vs iso_flag
    y_true = df["hard_wd"].values
    y_pred = df["iso_flag"].values
    if y_true.sum() > 0:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        log.write(f"\n-- window_dressing vs iso_flag --")
        log.write(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

    # agreement with stress
    ct = pd.crosstab(df["alert_level"], df["hard_stress"], margins=True)
    log.write(f"\n-- alert_level vs is_stressed --\n{ct}")

    return df


# ---------------------------------------------------------------
# Synthetic injection
# ---------------------------------------------------------------
def synthetic_anomaly_test(monthly, log, n_inject=100):
    log.write("\n" + "=" * 60)
    log.write(f"Synthetic anomaly injection ({n_inject} customers)")
    log.write("=" * 60)

    rng = np.random.RandomState(RANDOM_STATE)
    all_cust = monthly["customer_id"].unique()
    inject = rng.choice(all_cust, size=n_inject, replace=False)

    m = monthly.copy()
    for cid in inject:
        sub = m[m["customer_id"] == cid].sort_values("month")
        if len(sub) < 5:
            continue
        target_idx = sub.index[3:5]  # months 4 and 5
        m.loc[target_idx, "salary_sum"] = 0
        m.loc[target_idx, "savings_rate"] = -0.5
        m.loc[target_idx, "discretionary_sum"] = m.loc[target_idx, "discretionary_sum"] * 3

    # Re-run per-customer z-score detection
    feats = [c for c in MONTHLY_FEATURES if c in m.columns]
    df = m.sort_values(["customer_id", "month"]).copy()
    z_feats = []
    for f in feats:
        grp = df.groupby("customer_id")[f]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        col = f"z_{f}"
        df[col] = ((df[f] - mean) / std).fillna(0)
        z_feats.append(col)

    df["month_max_z"] = df[z_feats].abs().max(axis=1)
    df["month_is_anomaly"] = (df["month_max_z"] >= 2.5).astype(int)

    per_cust = df.groupby("customer_id")["month_is_anomaly"].sum().reset_index()
    per_cust["is_injected"] = per_cust["customer_id"].isin(inject).astype(int)
    per_cust["predicted"] = (per_cust["month_is_anomaly"] >= 2).astype(int)

    prec = precision_score(per_cust["is_injected"], per_cust["predicted"], zero_division=0)
    rec = recall_score(per_cust["is_injected"], per_cust["predicted"], zero_division=0)
    f1 = f1_score(per_cust["is_injected"], per_cust["predicted"], zero_division=0)

    log.write(f"Precision: {prec:.4f}")
    log.write(f"Recall:    {rec:.4f}")
    log.write(f"F1:        {f1:.4f}")

    return prec, rec, f1


# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------
def plot_alert_distribution(df, log):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["alert_level"].value_counts().reindex(["low", "medium", "high"]).fillna(0)
    axes[0].bar(counts.index, counts.values, color=["#2a9d8f", "#f4a261", "#e76f51"])
    axes[0].set_title("Alert level distribution")
    axes[0].set_ylabel("Customers")

    axes[1].hist(df["anomaly_months_count"], bins=20, color="#264653")
    axes[1].set_title("Anomalous months per customer")
    axes[1].set_xlabel("# anomalous months")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "anomaly_distribution.png", dpi=110)
    plt.close()
    log.write(f"Saved: {FIG_DIR / 'anomaly_distribution.png'}")


def plot_anomaly_vs_stress(df, log):
    fig, ax = plt.subplots(figsize=(8, 6))
    ct = pd.crosstab(df["alert_level"], df["hard_stress"], normalize="index")
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Stress rate by alert level")
    ax.set_xlabel("Alert level")
    ax.set_ylabel("Fraction with is_stressed=1")
    ax.legend(title="is_stressed", labels=["No", "Yes"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "anomaly_vs_stress.png", dpi=110)
    plt.close()
    log.write(f"Saved: {FIG_DIR / 'anomaly_vs_stress.png'}")


def plot_flags(df, log):
    high = df[df["alert_level"] == "high"]
    rates = high[["hard_bounce", "hard_income_gap", "hard_savings_drop",
                  "hard_wd", "hard_stress"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(rates.index, rates.values, color="#264653")
    ax.set_xlabel("Fraction of high-alert customers")
    ax.set_title("Which flags drive high alerts")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "anomaly_flags.png", dpi=110)
    plt.close()
    log.write(f"Saved: {FIG_DIR / 'anomaly_flags.png'}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 5 — Anomaly Detection (v2) — {datetime.now().isoformat(timespec='seconds')}")

    ledger = pd.read_parquet(INTERIM / "transaction_ledger.parquet")
    ledger["date"] = pd.to_datetime(ledger["date"])
    stress = pd.read_parquet(PROCESSED / "stress_features.parquet")

    monthly = build_monthly_with_june(log)
    df_monthly, per_cust, z_feats = per_customer_anomaly_features(monthly, log)
    iso, per_cust = isolation_forest_customer_level(df_monthly, per_cust, z_feats, log)

    flags = rule_hard_flags(ledger, monthly, stress, log)
    df = combine_alerts(per_cust, flags, log)
    df = eval_ground_truth(df, log)

    prec, rec, f1 = synthetic_anomaly_test(monthly, log)

    plot_alert_distribution(df, log)
    plot_anomaly_vs_stress(df, log)
    plot_flags(df, log)

    bundle = {
        "iso_forest": iso,
        "z_feats": z_feats,
        "monthly_features": [c for c in MONTHLY_FEATURES if c in monthly.columns],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "synthetic_test": {"precision": prec, "recall": rec, "f1": f1},
    }
    joblib.dump(bundle, MODELS_DIR / "anomaly.pkl")
    df.to_parquet(PROCESSED / "anomaly_features.parquet", index=False)
    log.write(f"\nSaved: {MODELS_DIR / 'anomaly.pkl'}")
    log.write(f"Saved: {PROCESSED / 'anomaly_features.parquet'}")

    log.write("\n" + "=" * 60)
    log.write("PHASE 5 SUMMARY")
    log.write("=" * 60)
    log.write(f"Alert distribution:\n{df['alert_level'].value_counts()}")
    log.write(f"\nSynthetic anomaly test — P/R/F1: {prec:.4f} / {rec:.4f} / {f1:.4f}")

    log.save(REPORT_DIR / "phase5_anomaly.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase5_anomaly.txt'}")


if __name__ == "__main__":
    main()
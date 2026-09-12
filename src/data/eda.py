"""
src/data/eda.py
Phase 1 - Exploratory Data Analysis.
Reads cleaned parquets, produces summary stats and charts.

Run from SamriddhiAI/ root:
    py -3.11 -m src.data.eda

Outputs:
    outputs/figures/*.png
    outputs/reports/phase1_eda_summary.txt
"""

from pathlib import Path
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

INTERIM_DIR = Path("data/interim")
FIG_DIR = Path("outputs/figures")
REPORT_DIR = Path("outputs/reports")


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def load_all():
    master = pd.read_parquet(INTERIM_DIR / "customer_master.parquet")
    labels = pd.read_parquet(INTERIM_DIR / "labels.parquet")
    ledger = pd.read_parquet(INTERIM_DIR / "transaction_ledger.parquet")
    behavior = pd.read_parquet(INTERIM_DIR / "behavioral_log.parquet")
    return master, labels, ledger, behavior


def safe_plot(name, fn, log):
    """Run a plot function; catch and log any error so others still run."""
    try:
        fn()
        log.write(f"  [OK] {name}")
    except Exception as e:
        log.write(f"  [FAIL] {name}: {e}")
        log.write(traceback.format_exc())


# ---------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------
def plot_target_balance(labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    applied_counts = labels["applied_flag"].value_counts().sort_index()
    axes[0].bar(["No (0)", "Yes (1)"], applied_counts.values, color=["#888", "#2a9d8f"])
    axes[0].set_title("applied_flag distribution")
    axes[0].set_ylabel("Customers")

    loan_counts = labels["loan_type"].value_counts(dropna=False)
    axes[1].bar(loan_counts.index.astype(str), loan_counts.values, color="#264653")
    axes[1].set_title("loan_type distribution")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_target_balance.png", dpi=110)
    plt.close()


def plot_demographics(master):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].hist(master["age"].dropna().astype(int), bins=20, color="#e76f51")
    axes[0, 0].set_title("Age distribution")

    occ = master["occupation_type"].value_counts()
    axes[0, 1].bar(occ.index.astype(str), occ.values, color="#2a9d8f")
    axes[0, 1].set_title("Occupation type")

    tier = master["city_tier"].value_counts()
    axes[1, 0].bar(tier.index.astype(str), tier.values, color="#264653")
    axes[1, 0].set_title("City tier")

    axes[1, 1].hist(master["declared_annual_income"].dropna(), bins=30, color="#f4a261")
    axes[1, 1].set_title("Declared annual income (INR)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_demographics.png", dpi=110)
    plt.close()


def plot_bureau_score(master):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(master["bureau_score_proxy"].dropna().astype(int), bins=30, color="#264653")
    ax.set_title("Bureau score proxy")
    ax.set_xlabel("Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_bureau_score.png", dpi=110)
    plt.close()


def plot_transaction_categories(ledger):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    cats = ledger["true_category"].value_counts()
    axes[0].bar(cats.index.astype(str), cats.values, color="#2a9d8f")
    axes[0].set_title("true_category counts")
    axes[0].tick_params(axis="x", rotation=45)

    chans = ledger["channel"].value_counts()
    axes[1].bar(chans.index.astype(str), chans.values, color="#e76f51")
    axes[1].set_title("Channel counts")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_transaction_categories.png", dpi=110)
    plt.close()


def plot_monthly_txn_volume(ledger):
    ledger = ledger.copy()
    ledger["month"] = ledger["date"].dt.to_period("M").dt.to_timestamp()
    monthly = ledger.groupby("month").size()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(monthly.index, monthly.values, marker="o", color="#264653")
    ax.set_title("Monthly transaction volume (all customers)")
    ax.set_ylabel("Transactions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_monthly_txn_volume.png", dpi=110)
    plt.close()


def plot_behavioral(behavior):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ev = behavior["event_type"].value_counts()
    axes[0].bar(ev.index.astype(str), ev.values, color="#264653")
    axes[0].set_title("Event type counts")
    axes[0].tick_params(axis="x", rotation=30)

    if "session_duration_seconds" in behavior.columns:
        axes[1].hist(
            behavior["session_duration_seconds"].dropna(),
            bins=40, color="#f4a261",
        )
        axes[1].set_title("Session duration (seconds)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_behavioral.png", dpi=110)
    plt.close()


def plot_narration_examples(ledger, log):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    lines = ["Sample narrations per true_category:\n"]
    for cat in sorted(ledger["true_category"].dropna().unique()):
        samples = ledger[ledger["true_category"] == cat]["narration_raw"].head(4).tolist()
        lines.append(f"\n{cat}:")
        for s in samples:
            s_short = (str(s)[:95] + "...") if len(str(s)) > 95 else str(s)
            lines.append(f"  - {s_short}")
    ax.text(0.01, 0.99, "\n".join(lines), va="top", family="monospace", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_narration_samples.png", dpi=110)
    plt.close()
    log.write("  (also see outputs/figures/07_narration_samples.png for raw examples)")


# ---------------------------------------------------------------
# Insights
# ---------------------------------------------------------------
def print_insights(master, labels, ledger, behavior, log):
    log.write("\n" + "=" * 60)
    log.write("EDA INSIGHTS")
    log.write("=" * 60)

    log.write(f"\nCustomers: {len(master)}")
    log.write(f"Applied rate: {labels['applied_flag'].mean():.4f}")
    log.write(f"Converted rate: {labels['converted_flag'].mean():.4f}")

    merged = master.merge(labels[["customer_id", "applied_flag"]], on="customer_id")

    log.write("\nApplied rate by city_tier:")
    log.write(str(merged.groupby("city_tier")["applied_flag"].mean().round(4)))

    log.write("\nApplied rate by occupation_type:")
    log.write(str(merged.groupby("occupation_type")["applied_flag"].mean().round(4)))

    corr = merged[["bureau_score_proxy", "applied_flag"]].dropna().corr().iloc[0, 1]
    log.write(f"\nBureau score correlation with applied_flag: {corr:.4f}")

    ledger2 = ledger.copy()
    ledger2["month"] = ledger2["date"].dt.to_period("M")
    txns_per_month = ledger2.groupby("month").size()
    log.write(f"\nTxns per month — mean: {txns_per_month.mean():.0f}, "
              f"min: {txns_per_month.min()}, max: {txns_per_month.max()}")
    log.write(f"Distinct months present: {ledger2['month'].nunique()}")

    log.write("\nSanity: null existing_loan_types vs existing_loan_count == 0")
    chk = master[master["existing_loan_types"].isnull()]["existing_loan_count"].value_counts()
    log.write(str(chk))

    log.write("\nwindow_dressing customers:")
    wd = ledger[ledger["true_category"] == "window_dressing"]["customer_id"].nunique()
    log.write(f"  {wd} unique customers have ≥1 window_dressing txn")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 1 EDA Report — {datetime.now().isoformat(timespec='seconds')}")

    master, labels, ledger, behavior = load_all()

    log.write("\nGenerating plots...")
    safe_plot("01_target_balance", lambda: plot_target_balance(labels), log)
    safe_plot("02_demographics", lambda: plot_demographics(master), log)
    safe_plot("03_bureau_score", lambda: plot_bureau_score(master), log)
    safe_plot("04_transaction_categories", lambda: plot_transaction_categories(ledger), log)
    safe_plot("05_monthly_txn_volume", lambda: plot_monthly_txn_volume(ledger), log)
    safe_plot("06_behavioral", lambda: plot_behavioral(behavior), log)
    safe_plot("07_narration_samples", lambda: plot_narration_examples(ledger, log), log)

    print_insights(master, labels, ledger, behavior, log)

    log.save(REPORT_DIR / "phase1_eda_summary.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase1_eda_summary.txt'}")
    print(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
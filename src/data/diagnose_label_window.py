"""
src/data/diagnose_label_window.py
Phase 3 Pre-Step — Determine whether labels describe future outcomes or
whole-period status. Detects leakage risk in transaction-based features.

Run from SamriddhiAI/ root:
    py -3.11 -m src.data.diagnose_label_window

Outputs:
    outputs/reports/phase3_prelabel_diagnosis.txt
    outputs/figures/prelabel_gap_over_time.png
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INTERIM = Path("data/interim")
REPORT_DIR = Path("outputs/reports")
FIG_DIR = Path("outputs/figures")


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def load():
    master = pd.read_parquet(INTERIM / "customer_master.parquet")
    labels = pd.read_parquet(INTERIM / "labels.parquet")
    ledger = pd.read_parquet(INTERIM / "transaction_ledger.parquet")
    return master, labels, ledger


def enrich_ledger(ledger):
    ledger = ledger.copy()
    ledger["month"] = ledger["date"].dt.to_period("M").dt.to_timestamp()
    return ledger


# ---------------------------------------------------------------
# Test 1 — Do applied and non-applied customers diverge over time?
# ---------------------------------------------------------------
def test_behavior_gap_over_time(ledger, labels, log):
    log.write("\n" + "=" * 60)
    log.write("TEST 1 — Behavioral gap between applied and non-applied customers over time")
    log.write("=" * 60)

    df = ledger.merge(labels[["customer_id", "applied_flag", "converted_flag"]],
                      on="customer_id", how="left")

    # Monthly credit sum per customer
    monthly = df.groupby(["customer_id", "applied_flag", "month"]).agg(
        credit_sum=("amount", lambda s: s[s > 0].sum()),
        debit_sum=("amount", lambda s: -s[s < 0].sum()),
        txn_count=("amount", "size"),
    ).reset_index()

    # Average per group per month
    gap = monthly.groupby(["month", "applied_flag"]).agg(
        avg_credit=("credit_sum", "mean"),
        avg_debit=("debit_sum", "mean"),
        avg_txn_count=("txn_count", "mean"),
    ).reset_index()

    pivot_credit = gap.pivot(index="month", columns="applied_flag", values="avg_credit")
    pivot_debit = gap.pivot(index="month", columns="applied_flag", values="avg_debit")
    pivot_txns = gap.pivot(index="month", columns="applied_flag", values="avg_txn_count")

    def pct_gap(p):
        return ((p[1] - p[0]) / p[0].abs().clip(lower=1) * 100).round(2)

    log.write("\nMonthly avg CREDIT (INR):")
    log.write(str(pivot_credit.round(0)))
    log.write("\n% gap (applied vs not applied):")
    log.write(str(pct_gap(pivot_credit)))

    log.write("\nMonthly avg DEBIT (INR):")
    log.write(str(pivot_debit.round(0)))
    log.write("\n% gap (applied vs not applied):")
    log.write(str(pct_gap(pivot_debit)))

    log.write("\nMonthly avg TXN COUNT:")
    log.write(str(pivot_txns.round(1)))

    # -------- Chart --------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(pivot_credit.index, pivot_credit[0], marker="o", label="Not applied")
    axes[0].plot(pivot_credit.index, pivot_credit[1], marker="o", label="Applied")
    axes[0].set_title("Monthly avg credit per customer")
    axes[0].set_ylabel("INR")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(pivot_debit.index, pivot_debit[0], marker="o", label="Not applied")
    axes[1].plot(pivot_debit.index, pivot_debit[1], marker="o", label="Applied")
    axes[1].set_title("Monthly avg debit per customer")
    axes[1].set_ylabel("INR")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "prelabel_gap_over_time.png", dpi=110)
    plt.close()
    log.write(f"\nChart saved: {FIG_DIR / 'prelabel_gap_over_time.png'}")

    return pct_gap(pivot_credit), pct_gap(pivot_debit)


# ---------------------------------------------------------------
# Test 2 — Do converted customers suddenly get new EMI debits later?
# ---------------------------------------------------------------
def test_new_emi_for_converted(ledger, labels, log):
    log.write("\n" + "=" * 60)
    log.write("TEST 2 — Do converted customers suddenly show new EMI activity?")
    log.write("=" * 60)

    df = ledger.merge(labels[["customer_id", "converted_flag", "loan_type"]],
                      on="customer_id", how="left")

    emi = df[df["true_category"] == "emi"].copy()
    emi_per_month = emi.groupby(["customer_id", "converted_flag", "month"]).size() \
                       .reset_index(name="emi_count")

    agg = emi_per_month.groupby(["month", "converted_flag"]).agg(
        avg_emi_count=("emi_count", "mean"),
        n_customers_with_emi=("customer_id", "nunique"),
    ).reset_index()

    pivot = agg.pivot(index="month", columns="converted_flag", values="avg_emi_count")
    log.write("\nAvg EMI txns per customer per month (only customers with ≥1 EMI):")
    log.write(str(pivot.round(2)))

    # Compare first 6 months vs last 6 months
    first_half = pivot.iloc[:6].mean()
    second_half = pivot.iloc[-6:].mean()
    log.write(f"\nFirst 6 months avg: converted={first_half[1]:.2f}  not={first_half[0]:.2f}")
    log.write(f"Last  6 months avg: converted={second_half[1]:.2f}  not={second_half[0]:.2f}")
    log.write(f"Growth ratio (converted): {(second_half[1] / first_half[1]):.2f}x")
    log.write(f"Growth ratio (not converted): {(second_half[0] / first_half[0]):.2f}x")

    if second_half[1] > first_half[1] * 1.15 and second_half[1] > second_half[0] * 1.15:
        log.write("\n>> Evidence of POST-application EMI activity in converted customers.")
        log.write(">> Interpretation: labels likely span the whole period, or a late cutoff.")
    else:
        log.write("\n>> No strong post-application EMI signature detected.")


# ---------------------------------------------------------------
# Test 3 — Any suspicious late credits for applied customers?
# ---------------------------------------------------------------
def test_late_credit_spikes(ledger, labels, log):
    log.write("\n" + "=" * 60)
    log.write("TEST 3 — Do applied customers show late-period credit spikes?")
    log.write("=" * 60)

    df = ledger.merge(labels[["customer_id", "applied_flag"]], on="customer_id", how="left")
    credits = df[df["direction"] == "credit"]

    per_cust_month = credits.groupby(["customer_id", "applied_flag", "month"])["amount"].sum() \
                            .reset_index()

    for flag, name in [(0, "not applied"), (1, "applied")]:
        sub = per_cust_month[per_cust_month["applied_flag"] == flag]
        first3 = sub[sub["month"].isin(sub["month"].unique()[:3])]["amount"].mean()
        last3 = sub[sub["month"].isin(sub["month"].unique()[-3:])]["amount"].mean()
        log.write(f"\n{name}: first-3-month avg credit = {first3:,.0f}, "
                  f"last-3-month avg credit = {last3:,.0f}, ratio = {last3/first3:.2f}x")


# ---------------------------------------------------------------
# Test 4 — Cross-tab: loan_type vs applied (sanity)
# ---------------------------------------------------------------
def test_loan_type_consistency(labels, log):
    log.write("\n" + "=" * 60)
    log.write("TEST 4 — loan_type consistency with applied_flag")
    log.write("=" * 60)
    ct = pd.crosstab(labels["applied_flag"], labels["loan_type"], margins=True)
    log.write(str(ct))


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 3 Pre-Step Diagnosis — {datetime.now().isoformat(timespec='seconds')}")
    log.write("Goal: decide whether labels describe future outcomes (safe) or whole-period")
    log.write("      status (leakage risk) before we engineer features.")

    master, labels, ledger = load()
    ledger = enrich_ledger(ledger)

    test_behavior_gap_over_time(ledger, labels, log)
    test_new_emi_for_converted(ledger, labels, log)
    test_late_credit_spikes(ledger, labels, log)
    test_loan_type_consistency(labels, log)

    log.write("\n" + "=" * 60)
    log.write("INTERPRETATION GUIDE")
    log.write("=" * 60)
    log.write("""
- If the applied/non-applied gap is roughly constant across months → labels are
  future-facing (or at least well-separated from features). Safe to use any
  snapshot month.

- If the applied group's debit/EMI activity GROWS in later months relative to the
  non-applied group → the label correlates with same-period behavior. We must
  restrict feature window to early months (e.g., 2024 only) to avoid leakage.

- If converted customers show a clear new EMI series in later months, this
  confirms the label is captured within the observation window and we should
  use pre-2025 features to avoid contamination.
""")

    log.save(REPORT_DIR / "phase3_prelabel_diagnosis.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase3_prelabel_diagnosis.txt'}")


if __name__ == "__main__":
    main()
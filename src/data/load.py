"""
src/data/load.py
Phase 1 - Load, validate, and save all 4 raw CSVs as parquet.

Run from SamriddhiAI/ root:
    py -3.11 -m src.data.load

Outputs:
    data/interim/customer_master.parquet
    data/interim/transaction_ledger.parquet
    data/interim/behavioral_log.parquet
    data/interim/labels.parquet
    outputs/reports/phase1_validation.txt
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
REPORT_DIR = Path("outputs/reports")

# ---------------------------------------------------------------
# Helper: collect validation messages so we can print + save them
# ---------------------------------------------------------------
class ValidationLog:
    def __init__(self):
        self.lines = []

    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


# ---------------------------------------------------------------
# Load each CSV with explicit dtypes where it matters
# ---------------------------------------------------------------
def load_customer_master(log: ValidationLog) -> pd.DataFrame:
    log.write("\n" + "=" * 60)
    log.write("LOADING customer_master.csv")
    log.write("=" * 60)

    df = pd.read_csv(RAW_DIR / "customer_master.csv")
    log.write(f"Shape: {df.shape}")
    log.write(f"Columns: {list(df.columns)}")

    # ---- dtypes ----
    df["customer_id"] = df["customer_id"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
    df["employment_tenure_months"] = pd.to_numeric(
        df["employment_tenure_months"], errors="coerce"
    ).astype("Int64")
    df["declared_annual_income"] = pd.to_numeric(
        df["declared_annual_income"], errors="coerce"
    )
    df["existing_loan_count"] = pd.to_numeric(
        df["existing_loan_count"], errors="coerce"
    ).astype("Int64")
    df["relationship_tenure_years"] = pd.to_numeric(
        df["relationship_tenure_years"], errors="coerce"
    )
    df["bureau_score_proxy"] = pd.to_numeric(
        df["bureau_score_proxy"], errors="coerce"
    ).astype("Int64")

    # ---- nulls ----
    nulls = df.isnull().sum()
    log.write(f"\nNull counts per column:\n{nulls[nulls > 0] if (nulls > 0).any() else '  (none)'}")

    # ---- duplicates ----
    dup_ids = df["customer_id"].duplicated().sum()
    log.write(f"\nDuplicate customer_id: {dup_ids}")

    # ---- categorical value checks ----
    for col in ["occupation_type", "city_tier", "residence_type"]:
        if col in df.columns:
            log.write(f"\n{col} value counts:\n{df[col].value_counts(dropna=False)}")

    # ---- range checks ----
    log.write(f"\nage range: {df['age'].min()} to {df['age'].max()}  (expected 21-60)")
    log.write(f"bureau_score_proxy range: {df['bureau_score_proxy'].min()} to {df['bureau_score_proxy'].max()}")
    log.write(f"declared_annual_income range: {df['declared_annual_income'].min():,.0f} to {df['declared_annual_income'].max():,.0f}")

    return df


def load_labels(log: ValidationLog) -> pd.DataFrame:
    log.write("\n" + "=" * 60)
    log.write("LOADING labels.csv")
    log.write("=" * 60)

    df = pd.read_csv(RAW_DIR / "labels.csv")
    log.write(f"Shape: {df.shape}")
    log.write(f"Columns: {list(df.columns)}")

    df["customer_id"] = df["customer_id"].astype(str)
    df["applied_flag"] = pd.to_numeric(df["applied_flag"], errors="coerce").astype("Int64")
    df["converted_flag"] = pd.to_numeric(df["converted_flag"], errors="coerce").astype("Int64")
    df["application_amount"] = pd.to_numeric(df["application_amount"], errors="coerce")

    log.write(f"\napplied_flag counts:\n{df['applied_flag'].value_counts(dropna=False)}")
    log.write(f"\nconverted_flag counts:\n{df['converted_flag'].value_counts(dropna=False)}")
    log.write(f"\nloan_type value counts:\n{df['loan_type'].value_counts(dropna=False)}")

    # ---- logical consistency: converted implies applied ----
    bad = df[(df["converted_flag"] == 1) & (df["applied_flag"] == 0)]
    log.write(f"\nRows where converted=1 but applied=0 (should be 0): {len(bad)}")

    dup_ids = df["customer_id"].duplicated().sum()
    log.write(f"Duplicate customer_id: {dup_ids}")

    return df


def load_transaction_ledger(log: ValidationLog) -> pd.DataFrame:
    log.write("\n" + "=" * 60)
    log.write("LOADING transaction_ledger.csv")
    log.write("=" * 60)

    df = pd.read_csv(RAW_DIR / "transaction_ledger.csv")
    log.write(f"Shape: {df.shape}")
    log.write(f"Columns: {list(df.columns)}")

    df["customer_id"] = df["customer_id"].astype(str)
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    log.write(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    log.write(f"Null dates: {df['date'].isnull().sum()}")
    log.write(f"Null amounts: {df['amount'].isnull().sum()}")

    log.write(f"\ndirection counts:\n{df['direction'].value_counts(dropna=False)}")
    log.write(f"\nchannel counts:\n{df['channel'].value_counts(dropna=False)}")
    log.write(f"\ntrue_category counts:\n{df['true_category'].value_counts(dropna=False)}")

    # ---- sign consistency check ----
    credit_neg = df[(df["direction"] == "credit") & (df["amount"] < 0)]
    debit_pos = df[(df["direction"] == "debit") & (df["amount"] > 0)]
    log.write(f"\nCredit rows with negative amount (should be 0): {len(credit_neg)}")
    log.write(f"Debit rows with positive amount (should be 0): {len(debit_pos)}")

    # ---- duplicate transaction_id ----
    dup_txn = df["transaction_id"].duplicated().sum()
    log.write(f"Duplicate transaction_id: {dup_txn}")

    # ---- txns per customer ----
    per_cust = df.groupby("customer_id").size()
    log.write(f"\nTxns per customer:\n{per_cust.describe()}")

    log.write(f"\nSample narrations per category:")
    for cat in df["true_category"].dropna().unique():
        sample = df[df["true_category"] == cat]["narration_raw"].head(2).tolist()
        log.write(f"  {cat}: {sample}")

    return df


def load_behavioral_log(log: ValidationLog) -> pd.DataFrame:
    log.write("\n" + "=" * 60)
    log.write("LOADING behavioral_log.csv")
    log.write("=" * 60)

    df = pd.read_csv(RAW_DIR / "behavioral_log.csv")
    log.write(f"Shape: {df.shape}")
    log.write(f"Columns: {list(df.columns)}")

    df["customer_id"] = df["customer_id"].astype(str)
    df["event_id"] = df["event_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "session_duration_seconds" in df.columns:
        df["session_duration_seconds"] = pd.to_numeric(
            df["session_duration_seconds"], errors="coerce"
        )

    log.write(f"\nTimestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    log.write(f"Null timestamps: {df['timestamp'].isnull().sum()}")

    log.write(f"\nevent_type counts:\n{df['event_type'].value_counts(dropna=False)}")
    if "product_viewed" in df.columns:
        log.write(f"\nproduct_viewed counts:\n{df['product_viewed'].value_counts(dropna=False)}")

    dup_evt = df["event_id"].duplicated().sum()
    log.write(f"\nDuplicate event_id: {dup_evt}")

    events_per_cust = df.groupby("customer_id").size()
    log.write(f"\nEvents per customer:\n{events_per_cust.describe()}")

    return df


# ---------------------------------------------------------------
# Cross-table referential integrity
# ---------------------------------------------------------------
def check_referential_integrity(master, labels, ledger, behavior, log: ValidationLog):
    log.write("\n" + "=" * 60)
    log.write("CROSS-TABLE REFERENTIAL INTEGRITY")
    log.write("=" * 60)

    master_ids = set(master["customer_id"])
    log.write(f"Unique customers in master: {len(master_ids)}")

    for name, df in [("labels", labels), ("ledger", ledger), ("behavior", behavior)]:
        ids = set(df["customer_id"])
        orphans = ids - master_ids
        log.write(f"\n{name}: {len(ids)} unique customers")
        log.write(f"  Orphans (in {name} but not in master): {len(orphans)}")
        if orphans:
            log.write(f"  Sample orphans: {list(orphans)[:5]}")

    # How many master customers appear in each event table
    for name, df in [("ledger", ledger), ("behavior", behavior)]:
        present = len(set(df["customer_id"]) & master_ids)
        log.write(f"\n{name}: {present}/{len(master_ids)} master customers have events "
                  f"({100 * present / len(master_ids):.1f}%)")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = ValidationLog()
    log.write(f"Phase 1 Validation Report")
    log.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}")

    master = load_customer_master(log)
    labels = load_labels(log)
    ledger = load_transaction_ledger(log)
    behavior = load_behavioral_log(log)

    check_referential_integrity(master, labels, ledger, behavior, log)

    # ---- Save parquets ----
    log.write("\n" + "=" * 60)
    log.write("SAVING CLEANED PARQUETS")
    log.write("=" * 60)
    master.to_parquet(INTERIM_DIR / "customer_master.parquet", index=False)
    labels.to_parquet(INTERIM_DIR / "labels.parquet", index=False)
    ledger.to_parquet(INTERIM_DIR / "transaction_ledger.parquet", index=False)
    behavior.to_parquet(INTERIM_DIR / "behavioral_log.parquet", index=False)
    log.write(f"Saved 4 parquet files to {INTERIM_DIR}/")

    log.save(REPORT_DIR / "phase1_validation.txt")
    print(f"\nValidation report saved to {REPORT_DIR / 'phase1_validation.txt'}")


if __name__ == "__main__":
    main()
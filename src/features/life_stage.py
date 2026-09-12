"""
src/features/life_stage.py
Phase 3.4 - Rule-based life-stage segmentation.

Output:
    data/processed/life_stage_features.parquet  (customer_id, ~10 cols)
    outputs/reports/phase3_4_life_stage.txt

Run from SamriddhiAI/ root:
    py -3.11 -m src.features.life_stage
"""

from pathlib import Path
from datetime import datetime
import pandas as pd

PROCESSED = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

STAGES = [
    "debt_stressed",
    "mortgage_holder",
    "young_renter",
    "young_owner_no_debt",
    "young_family",
    "established_borrower",
    "established_saver",
    "pre_retiree",
    "unclassified",
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


def load_inputs(log):
    profile = pd.read_parquet(PROCESSED / "profile_features.parquet")
    txn_snap = pd.read_parquet(PROCESSED / "txn_snapshot.parquet")

    # Merge to one frame — one row per customer
    df = profile.merge(txn_snap, on="customer_id", how="left", suffixes=("", "_txn"))
    log.write(f"Loaded and merged: {df.shape}")
    return df


def check_columns(df, log):
    """Verify the columns we need exist; warn if missing."""
    needed = [
        "age", "residence_owned", "has_home_loan", "has_mortgage_loan",
        "existing_loan_count", "has_any_existing_loan",
        "savings_rate_mean_7", "emi_to_income_mean_7",
        "has_bounce", "missed_emi_in_window",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        log.write(f"\n[WARN] Missing columns: {missing}")
    else:
        log.write("\nAll required columns present.")
    return needed


def assign_stage(row):
    """Return the highest-priority matching life stage."""
    age = row["age"]
    owned = row["residence_owned"]
    has_mortgage = (row["has_home_loan"] == 1) or (row["has_mortgage_loan"] == 1)
    loan_count = row["existing_loan_count"] if pd.notna(row["existing_loan_count"]) else 0
    savings = row["savings_rate_mean_7"]
    emi_inc = row["emi_to_income_mean_7"]
    bounce = row["has_bounce"]
    missed = row["missed_emi_in_window"]

    # 1. debt_stressed (highest priority)
    if (emi_inc > 0.4) or (bounce == 1) or (missed > 0):
        return "debt_stressed"

    # 2. mortgage_holder
    if has_mortgage:
        return "mortgage_holder"

    # 3. young_renter
    if (age <= 30) and (owned == 0):
        return "young_renter"

    # 4. young_owner_no_debt (NEW)
    if (age <= 30) and (owned == 1) and (loan_count == 0):
        return "young_owner_no_debt"

    # 5. young_family
    if (30 < age <= 45) and (loan_count == 0):
        return "young_family"

    # 6. established_borrower (NEW) — has non-mortgage loans, not stressed
    if (loan_count >= 1) and (age <= 50):
        return "established_borrower"

    # 7. established_saver
    if (age > 45) and (loan_count == 0) and (savings > 0.15):
        return "established_saver"

    # 8. pre_retiree
    if (age > 50) and (loan_count <= 1):
        return "pre_retiree"

    return "unclassified"


def build_life_stage(df, log):
    log.write("\n" + "=" * 60)
    log.write("Assigning life stages")
    log.write("=" * 60)

    # Handle missing numeric columns gracefully
    for col in ["savings_rate_mean_7", "emi_to_income_mean_7"]:
        if col not in df.columns:
            df[col] = 0.0
    for col in ["has_bounce", "missed_emi_in_window"]:
        if col not in df.columns:
            df[col] = 0

    df["life_stage"] = df.apply(assign_stage, axis=1)

    log.write(f"life_stage value counts:\n{df['life_stage'].value_counts()}")

    # One-hot
    for stage in STAGES:
        df[f"stage_{stage}"] = (df["life_stage"] == stage).astype(int)

    # Loan target flag — not debt stressed, not unclassified
    df["is_loan_target"] = (
        (~df["life_stage"].isin(["debt_stressed", "unclassified"]))
    ).astype(int)

    log.write(f"\nis_loan_target count: {df['is_loan_target'].sum()} "
              f"({100 * df['is_loan_target'].mean():.1f}%)")

    keep = ["customer_id", "life_stage", "is_loan_target"] + \
           [f"stage_{s}" for s in STAGES]
    out = df[keep].copy()

    log.write(f"\nFinal shape: {out.shape}")
    return out


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 3.4 — Life Stage — {datetime.now().isoformat(timespec='seconds')}")

    df = load_inputs(log)
    check_columns(df, log)
    out = build_life_stage(df, log)

    out.to_parquet(PROCESSED / "life_stage_features.parquet", index=False)
    log.write(f"\nSaved: {PROCESSED / 'life_stage_features.parquet'}")

    log.save(REPORT_DIR / "phase3_4_life_stage.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase3_4_life_stage.txt'}")


if __name__ == "__main__":
    main()
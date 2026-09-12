"""
src/features/profile_features.py
Phase 3.1 - Build profile features from customer_master.parquet.

One row per customer. Includes raw numerics, log-transformed income,
loan-type flags parsed from '|'-separated existing_loan_types,
and one-hot encodings of occupation/city_tier/residence.

Run from SamriddhiAI/ root:
    py -3.11 -m src.features.profile_features

Outputs:
    data/processed/profile_features.parquet
    outputs/reports/phase3_1_profile.txt
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

LOAN_TYPES = ["home", "auto", "personal", "mortgage"]


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def load_master(log):
    df = pd.read_parquet(INTERIM / "customer_master.parquet")
    log.write(f"Loaded customer_master: {df.shape}")
    return df


def parse_existing_loans(df, log):
    """Split '|'-separated loan types into binary flags."""
    log.write("\nParsing existing_loan_types...")

    # Fill nulls with empty string
    types = df["existing_loan_types"].fillna("").astype(str).str.lower()

    for lt in LOAN_TYPES:
        df[f"has_{lt}_loan"] = types.str.contains(lt, regex=False).astype(int)

    df["has_any_existing_loan"] = (df["existing_loan_count"].fillna(0) > 0).astype(int)

    log.write(f"  has_home_loan count:     {df['has_home_loan'].sum()}")
    log.write(f"  has_auto_loan count:     {df['has_auto_loan'].sum()}")
    log.write(f"  has_personal_loan count: {df['has_personal_loan'].sum()}")
    log.write(f"  has_mortgage_loan count: {df['has_mortgage_loan'].sum()}")
    log.write(f"  has_any_existing_loan:   {df['has_any_existing_loan'].sum()}")

    return df


def add_bands(df, log):
    log.write("\nAdding band features...")

    # Age bands
    def age_band(a):
        if pd.isna(a): return "unknown"
        if a <= 30: return "21_30"
        if a <= 40: return "31_40"
        if a <= 50: return "41_50"
        return "51_60"
    df["age_band"] = df["age"].apply(age_band)

    # Bureau bands
    def bureau_band(b):
        if pd.isna(b): return "unknown"
        if b < 600: return "lt_600"
        if b < 700: return "600_699"
        if b < 800: return "700_799"
        return "gte_800"
    df["bureau_band"] = df["bureau_score_proxy"].apply(bureau_band)

    log.write(f"  age_band value counts:\n{df['age_band'].value_counts()}")
    log.write(f"  bureau_band value counts:\n{df['bureau_band'].value_counts()}")
    return df


def add_transforms(df, log):
    log.write("\nAdding transforms (log income, monthly income)...")
    df["monthly_declared_income"] = df["declared_annual_income"] / 12.0
    df["log_income"] = np.log1p(df["declared_annual_income"].clip(lower=0))
    return df


def add_one_hots(df, log):
    log.write("\nAdding one-hot encodings...")

    # Occupation
    df["occ_salaried"] = (df["occupation_type"] == "salaried").astype(int)
    df["occ_self_emp"] = (df["occupation_type"] == "self_employed").astype(int)
    df["occ_business"] = (df["occupation_type"] == "business_owner").astype(int)

    # City tier
    df["tier_metro"] = (df["city_tier"] == "metro").astype(int)
    df["tier_2"] = (df["city_tier"] == "tier2").astype(int)
    df["tier_3"] = (df["city_tier"] == "tier3").astype(int)

    # Residence
    df["residence_owned"] = (df["residence_type"] == "owned").astype(int)

    return df


def select_final_columns(df, log):
    """Keep the columns we want in the final feature table."""
    keep = [
        "customer_id",
        "age",
        "age_band",
        "employment_tenure_months",
        "relationship_tenure_years",
        "declared_annual_income",
        "log_income",
        "monthly_declared_income",
        "existing_loan_count",
        "has_any_existing_loan",
        "has_home_loan",
        "has_auto_loan",
        "has_personal_loan",
        "has_mortgage_loan",
        "bureau_score_proxy",
        "bureau_band",
        "occ_salaried",
        "occ_self_emp",
        "occ_business",
        "tier_metro",
        "tier_2",
        "tier_3",
        "residence_owned",
    ]
    out = df[keep].copy()

    log.write(f"\nFinal profile_features shape: {out.shape}")
    log.write(f"Columns: {list(out.columns)}")
    log.write(f"\nNull counts:\n{out.isnull().sum()[out.isnull().sum() > 0]}")
    log.write(f"\nDtypes:\n{out.dtypes}")
    return out


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 3.1 — Profile Features — {datetime.now().isoformat(timespec='seconds')}")

    master = load_master(log)
    master = parse_existing_loans(master, log)
    master = add_bands(master, log)
    master = add_transforms(master, log)
    master = add_one_hots(master, log)
    out = select_final_columns(master, log)

    out.to_parquet(PROCESSED / "profile_features.parquet", index=False)
    log.write(f"\nSaved: {PROCESSED / 'profile_features.parquet'}")

    log.save(REPORT_DIR / "phase3_1_profile.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase3_1_profile.txt'}")


if __name__ == "__main__":
    main()
    
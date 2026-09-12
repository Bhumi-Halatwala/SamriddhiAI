"""
src/features/build_model_input.py
Phase 3.6 - Merge all feature tables into the final model input.

Outputs:
    data/processed/model_input.parquet    (customer_id + all features)
    data/processed/labels_joined.parquet  (customer_id + labels, kept separate)
    outputs/reports/phase3_6_merge.txt

Run from SamriddhiAI/ root:
    py -3.11 -m src.features.build_model_input
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
INTERIM = Path("data/interim")
REPORT_DIR = Path("outputs/reports")

# Categorical string columns we drop (one-hots kept instead)
DROP_CATEGORICALS = [
    "life_stage",
    "dominant_product",
    "age_band",
    "bureau_band",
    "funnel_stage",
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


def load_all(log):
    log.write("\n" + "=" * 60)
    log.write("Loading feature tables")
    log.write("=" * 60)

    profile = pd.read_parquet(PROCESSED / "profile_features.parquet")
    txn = pd.read_parquet(PROCESSED / "txn_snapshot.parquet")
    behavior = pd.read_parquet(PROCESSED / "behavior_features.parquet")
    life = pd.read_parquet(PROCESSED / "life_stage_features.parquet")
    stress = pd.read_parquet(PROCESSED / "stress_features.parquet")
    labels = pd.read_parquet(INTERIM / "labels.parquet")

    log.write(f"profile:  {profile.shape}")
    log.write(f"txn:      {txn.shape}")
    log.write(f"behavior: {behavior.shape}")
    log.write(f"life:     {life.shape}")
    log.write(f"stress:   {stress.shape}")
    log.write(f"labels:   {labels.shape}")

    return profile, txn, behavior, life, stress, labels


def merge_features(profile, txn, behavior, life, stress, log):
    log.write("\n" + "=" * 60)
    log.write("Merging features")
    log.write("=" * 60)

    df = profile.copy()
    log.write(f"Start: {df.shape}")

    for name, other in [
        ("txn", txn),
        ("behavior", behavior),
        ("life", life),
        ("stress", stress),
    ]:
        # Drop customer_id before merging (already in df)
        other = other.drop(columns=["customer_id"], errors="ignore")
        # Check for column-name collisions
        overlap = set(df.columns) & set(other.columns)
        if overlap:
            log.write(f"  [WARN] {name}: overlapping columns {overlap}")
        df = df.merge(other, left_index=False, right_index=False,
                      left_on="customer_id", right_on=tc_idx(other))
        # Actually the above is wrong; do it simply below
    return df


def tc_idx(other):
    """Placeholder to avoid the buggy merge above — not used."""
    return None


def merge_features_clean(profile, txn, behavior, life, stress, log):
    """Clean merge: join on customer_id, drop dup key column."""
    log.write("\n" + "=" * 60)
    log.write("Merging features (clean)")
    log.write("=" * 60)

    frames = [
        ("profile", profile),
        ("txn", txn),
        ("behavior", behavior),
        ("life", life),
        ("stress", stress),
    ]

    df = None
    for name, f in frames:
        if df is None:
            df = f.copy()
            log.write(f"  {name}: {df.shape}")
            continue
        f = f.drop(columns=["customer_id"], errors="ignore")
        overlap = [c for c in f.columns if c in df.columns]
        if overlap:
            log.write(f"  [WARN] {name}: dropping overlapping columns {overlap}")
            f = f.drop(columns=overlap)
        df = pd.concat([df.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
        log.write(f"  + {name}: {df.shape}")

    log.write(f"\nMerged shape: {df.shape}")
    return df


def drop_constants(df, log):
    log.write("\n" + "=" * 60)
    log.write("Dropping constant columns (std == 0)")
    log.write("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constants = []
    for col in numeric_cols:
        if df[col].nunique(dropna=False) <= 1:
            constants.append(col)

    if constants:
        log.write(f"Dropping {len(constants)} constant columns:")
        for c in constants:
            log.write(f"  - {c}")
        df = df.drop(columns=constants)
    else:
        log.write("No constant columns found.")

    log.write(f"\nShape after dropping constants: {df.shape}")
    return df


def drop_categoricals(df, log):
    log.write("\n" + "=" * 60)
    log.write("Dropping raw categorical strings (one-hots retained)")
    log.write("=" * 60)

    present = [c for c in DROP_CATEGORICALS if c in df.columns]
    if present:
        log.write(f"Dropping: {present}")
        df = df.drop(columns=present)
    else:
        log.write("None present.")

    log.write(f"\nFinal shape: {df.shape}")
    return df


def build_labels(labels, log):
    log.write("\n" + "=" * 60)
    log.write("Preparing labels table")
    log.write("=" * 60)

    out = labels.copy()
    out["has_application_label"] = out["applied_flag"].notna().astype(int)

    log.write(f"Labels shape: {out.shape}")
    log.write(f"applied_flag mean: {out['applied_flag'].mean():.4f}")
    log.write(f"converted_flag mean: {out['converted_flag'].mean():.4f}")
    log.write(f"loan_type value counts:\n{out['loan_type'].value_counts(dropna=False)}")

    return out


def validate_no_leakage(model_input, log):
    log.write("\n" + "=" * 60)
    log.write("Leakage check: no label-like columns in model input")
    log.write("=" * 60)

    suspicious = [
        "applied_flag", "converted_flag", "loan_type",
        "application_amount", "has_application_label",
    ]
    found = [c for c in suspicious if c in model_input.columns]
    if found:
        log.write(f"[FAIL] Found label columns in model input: {found}")
    else:
        log.write("PASS: no label columns present in model_input.")
    return found


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log = Log()
    log.write(f"Phase 3.6 — Merge Model Input — {datetime.now().isoformat(timespec='seconds')}")

    profile, txn, behavior, life, stress, labels = load_all(log)
    df = merge_features_clean(profile, txn, behavior, life, stress, log)
    df = drop_constants(df, log)
    df = drop_categoricals(df, log)

    labels_df = build_labels(labels, log)
    bad = validate_no_leakage(df, log)

    # Save
    df.to_parquet(PROCESSED / "model_input.parquet", index=False)
    labels_df.to_parquet(PROCESSED / "labels_joined.parquet", index=False)
    log.write(f"\nSaved: {PROCESSED / 'model_input.parquet'}")
    log.write(f"Saved: {PROCESSED / 'labels_joined.parquet'}")

    # Final summary
    log.write("\n" + "=" * 60)
    log.write("SUMMARY")
    log.write("=" * 60)
    log.write(f"model_input shape: {df.shape}")
    log.write(f"labels_joined shape: {labels_df.shape}")
    log.write(f"Customer coverage: {df['customer_id'].nunique()} unique customers")
    log.write(f"Leakage check: {'FAIL' if bad else 'PASS'}")
    log.write(f"\nAll feature columns:")
    for i, c in enumerate(df.columns, 1):
        log.write(f"  {i:3d}. {c}")

    log.save(REPORT_DIR / "phase3_6_merge.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase3_6_merge.txt'}")


if __name__ == "__main__":
    main()
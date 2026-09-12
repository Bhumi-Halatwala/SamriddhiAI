"""
tests/test_features.py
Phase 3 tests — verify all feature tables and the merged model input.

Run from SamriddhiAI/ root:
    py -3.11 -m pytest tests/test_features.py -v
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

PROCESSED = Path("data/processed")
INTERIM = Path("data/interim")

EXPECTED_CUSTOMERS = 5000


# ---------------------------------------------------------------
# Fixtures (loaded once per session)
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def profile():
    return pd.read_parquet(PROCESSED / "profile_features.parquet")

@pytest.fixture(scope="module")
def txn_monthly():
    return pd.read_parquet(PROCESSED / "txn_monthly.parquet")

@pytest.fixture(scope="module")
def txn_snapshot():
    return pd.read_parquet(PROCESSED / "txn_snapshot.parquet")

@pytest.fixture(scope="module")
def behavior():
    return pd.read_parquet(PROCESSED / "behavior_features.parquet")

@pytest.fixture(scope="module")
def life():
    return pd.read_parquet(PROCESSED / "life_stage_features.parquet")

@pytest.fixture(scope="module")
def stress():
    return pd.read_parquet(PROCESSED / "stress_features.parquet")

@pytest.fixture(scope="module")
def model_input():
    return pd.read_parquet(PROCESSED / "model_input.parquet")

@pytest.fixture(scope="module")
def labels():
    return pd.read_parquet(PROCESSED / "labels_joined.parquet")


# ---------------------------------------------------------------
# Profile features
# ---------------------------------------------------------------
def test_profile_shape(profile):
    assert profile.shape[0] == EXPECTED_CUSTOMERS

def test_profile_unique_ids(profile):
    assert profile["customer_id"].is_unique

def test_profile_one_hots_sum_to_one(profile):
    occ = profile[["occ_salaried", "occ_self_emp", "occ_business"]].sum(axis=1)
    assert (occ == 1).all(), "occupation one-hots should sum to 1"
    tier = profile[["tier_metro", "tier_2", "tier_3"]].sum(axis=1)
    assert (tier == 1).all(), "tier one-hots should sum to 1"

def test_profile_log_income_positive(profile):
    assert (profile["log_income"] > 0).all()

def test_profile_loan_count_range(profile):
    assert profile["existing_loan_count"].between(0, 4).all()


# ---------------------------------------------------------------
# Transaction monthly
# ---------------------------------------------------------------
def test_txn_monthly_shape(txn_monthly):
    # 5000 customers, ~17 months → ~85000 rows
    assert txn_monthly.shape[0] == 85000, f"Expected 85000, got {txn_monthly.shape[0]}"

def test_txn_monthly_covers_all_customers(txn_monthly):
    assert txn_monthly["customer_id"].nunique() == EXPECTED_CUSTOMERS

def test_txn_monthly_months(txn_monthly):
    months = pd.to_datetime(txn_monthly["month"]).dt.to_period("M").nunique()
    assert months == 17, f"Expected 17 months, got {months}"

def test_txn_monthly_no_june_2025(txn_monthly):
    months = pd.to_datetime(txn_monthly["month"]).dt.to_period("M")
    assert "2025-06" not in months.astype(str).values, "June 2025 should be excluded"


# ---------------------------------------------------------------
# Transaction snapshot
# ---------------------------------------------------------------
def test_txn_snapshot_shape(txn_snapshot):
    assert txn_snapshot.shape[0] == EXPECTED_CUSTOMERS

def test_txn_snapshot_unique(txn_snapshot):
    assert txn_snapshot["customer_id"].is_unique

def test_txn_snapshot_no_nulls(txn_snapshot):
    nulls = txn_snapshot.isnull().sum().sum()
    assert nulls == 0, f"Expected 0 nulls, got {nulls}"


# ---------------------------------------------------------------
# Behavior features
# ---------------------------------------------------------------
def test_behavior_shape(behavior):
    assert behavior.shape[0] == EXPECTED_CUSTOMERS

def test_behavior_unique(behavior):
    assert behavior["customer_id"].is_unique

def test_behavior_no_nulls(behavior):
    assert behavior.isnull().sum().sum() == 0

def test_behavior_funnel_segments(behavior):
    # Should not be 100% one segment
    n_intent = behavior["funnel_application_intent"].sum()
    assert 500 < n_intent < 4500, f"funnel_application_intent = {n_intent} — too skewed"


# ---------------------------------------------------------------
# Life stage
# ---------------------------------------------------------------
def test_life_stage_shape(life):
    assert life.shape[0] == EXPECTED_CUSTOMERS

def test_life_stage_one_hot_sums_to_one(life):
    stage_cols = [c for c in life.columns if c.startswith("stage_")]
    sums = life[stage_cols].sum(axis=1)
    assert (sums == 1).all(), "life stage one-hots must sum to 1"

def test_life_stage_unclassified_low(life):
    n_unclass = life["stage_unclassified"].sum()
    assert n_unclass < 250, f"Too many unclassified: {n_unclass}"

def test_life_stage_is_loan_target_exists(life):
    assert "is_loan_target" in life.columns
    assert life["is_loan_target"].between(0, 1).all()


# ---------------------------------------------------------------
# Stress features
# ---------------------------------------------------------------
def test_stress_shape(stress):
    assert stress.shape[0] == EXPECTED_CUSTOMERS

def test_stress_buckets(stress):
    counts = stress["stress_level"].value_counts()
    assert set(counts.index) == {"low", "medium", "high"}
    # Percentile-based: 40 / 35 / 25
    assert counts["low"] == 2000
    assert counts["medium"] == 1750
    assert counts["high"] == 1250

def test_stress_score_range(stress):
    # After winsorization, no wild outliers
    assert stress["stress_score"].abs().max() < 5.0

def test_stress_bounce_correlation(stress, txn_snapshot):
    """The high-stress bucket should have meaningfully higher bounce rate."""
    merged = stress.merge(
        txn_snapshot[["customer_id", "has_bounce"]], on="customer_id"
    )
    rate_by_level = merged.groupby("stress_level")["has_bounce"].mean()
    assert rate_by_level["high"] > 0.35, \
        f"High-stress bucket bounce rate too low: {rate_by_level['high']}"
    assert rate_by_level["high"] > rate_by_level["low"] * 2, \
        "High-stress bounce rate should be >2x low-stress"


# ---------------------------------------------------------------
# Model input
# ---------------------------------------------------------------
def test_model_input_shape(model_input):
    assert model_input.shape[0] == EXPECTED_CUSTOMERS
    assert model_input.shape[1] >= 150, "Expected at least 150 features"

def test_model_input_unique(model_input):
    assert model_input["customer_id"].is_unique

def test_model_input_no_label_columns(model_input):
    forbidden = ["applied_flag", "converted_flag", "loan_type",
                 "application_amount", "has_application_label"]
    present = [c for c in forbidden if c in model_input.columns]
    assert not present, f"Label columns leaked into model_input: {present}"

def test_model_input_no_all_null_columns(model_input):
    all_null = [c for c in model_input.columns if model_input[c].isnull().all()]
    assert not all_null, f"All-null columns: {all_null}"

def test_model_input_numeric_only_except_id_and_stress_level(model_input):
    non_numeric = model_input.select_dtypes(exclude=["number"]).columns.tolist()
    expected = {"customer_id", "stress_level"}
    assert set(non_numeric).issubset(expected), \
        f"Unexpected non-numeric columns: {set(non_numeric) - expected}"


# ---------------------------------------------------------------
# Labels
# ---------------------------------------------------------------
def test_labels_shape(labels):
    assert labels.shape[0] == EXPECTED_CUSTOMERS

def test_labels_join(model_input, labels):
    assert set(labels["customer_id"]) == set(model_input["customer_id"])

def test_labels_converted_implies_applied(labels):
    bad = labels[(labels["converted_flag"] == 1) & (labels["applied_flag"] == 0)]
    assert len(bad) == 0


# ---------------------------------------------------------------
# Cross-table coverage
# ---------------------------------------------------------------
def test_all_customers_have_features(profile, txn_snapshot, behavior, life, stress):
    ids = set(profile["customer_id"])
    assert set(txn_snapshot["customer_id"]) == ids
    assert set(behavior["customer_id"]) == ids
    assert set(life["customer_id"]) == ids
    assert set(stress["customer_id"]) == ids
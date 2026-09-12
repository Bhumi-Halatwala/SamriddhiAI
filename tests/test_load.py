"""
tests/test_load.py
Basic sanity tests for Phase 1 outputs.
Run from SamriddhiAI/ root:
    py -3.11 -m pytest tests/test_load.py -v
"""

from pathlib import Path
import pandas as pd
import pytest

INTERIM = Path("data/interim")


@pytest.fixture(scope="module")
def master():
    return pd.read_parquet(INTERIM / "customer_master.parquet")


@pytest.fixture(scope="module")
def labels():
    return pd.read_parquet(INTERIM / "labels.parquet")


@pytest.fixture(scope="module")
def ledger():
    return pd.read_parquet(INTERIM / "transaction_ledger.parquet")


@pytest.fixture(scope="module")
def behavior():
    return pd.read_parquet(INTERIM / "behavioral_log.parquet")


def test_master_shape(master):
    assert master.shape[0] == 5000, f"Expected 5000 customers, got {master.shape[0]}"


def test_master_unique_ids(master):
    assert master["customer_id"].is_unique


def test_labels_shape(labels):
    assert labels.shape[0] == 5000


def test_labels_join_to_master(master, labels):
    assert set(labels["customer_id"]).issubset(set(master["customer_id"]))


def test_no_converted_without_applied(labels):
    bad = labels[(labels["converted_flag"] == 1) & (labels["applied_flag"] == 0)]
    assert len(bad) == 0


def test_ledger_dates_parsed(ledger):
    assert pd.api.types.is_datetime64_any_dtype(ledger["date"])


def test_ledger_no_null_amounts(ledger):
    assert ledger["amount"].isnull().sum() == 0


def test_ledger_sign_consistency(ledger):
    credit_neg = ((ledger["direction"] == "credit") & (ledger["amount"] < 0)).sum()
    debit_pos = ((ledger["direction"] == "debit") & (ledger["amount"] > 0)).sum()
    assert credit_neg == 0 and debit_pos == 0


def test_ledger_all_customers_in_master(master, ledger):
    assert set(ledger["customer_id"]).issubset(set(master["customer_id"]))


def test_behavior_all_customers_in_master(master, behavior):
    assert set(behavior["customer_id"]).issubset(set(master["customer_id"]))


def test_ledger_has_all_categories(ledger):
    expected = {"salary", "business_income", "emi", "rent", "discretionary", "bounce"}
    actual = set(ledger["true_category"].dropna().unique())
    missing = expected - actual
    assert not missing, f"Missing categories: {missing}"
"""
tests/test_narration.py
Phase 2 tests for the narration parser.
Run from SamriddhiAI/ root:
    py -3.11 -m pytest tests/test_narration.py -v
"""

from pathlib import Path
import pytest
from src.models.predict_narration import (
    predict_one,
    predict_categories,
    rule_based,
)


def test_model_file_exists():
    assert Path("models/narration_clf.pkl").exists()


# ---- each of these should be classified correctly ----
CASES = {
    "NEFT/416618/SAL/BHARATRETA": "salary",
    "NEFT-SALARY-NEXGENSYSTEMS-FEB2024": "salary",
    "ECS-EMI-ICICI-LOAN269641": "emi",
    "ECS-EMI-BANK-HOMELOAN-2025": "emi",
    "NEFT-RENT-LANDLORD-JAN2024": "rent",
    "UPI-SWIGGY-PAYMENT-41682": "discretionary",
    "UPI-BIGBASKET-PAYMENT-93123": "discretionary",
    "ECS-RETURN-INSUFFICIENT FUNDS": "bounce",
    "NEFT-BUSINESS RECEIPT-070124": "business_income",
    "NEFT-TRANSFER-IN-230625": "suspicious",
}


@pytest.mark.parametrize("text,expected", CASES.items())
def test_predictions(text, expected):
    pred = predict_one(text)
    assert pred == expected, f"'{text}' -> got {pred}, expected {expected}"


def test_batch_returns_correct_shapes():
    texts = list(CASES.keys())
    preds, confs, used = predict_categories(texts)
    assert len(preds) == len(texts)
    assert len(confs) == len(texts)
    assert len(used) == len(texts)


def test_rule_based_direct():
    assert rule_based("NEFT/XYZ/SAL/ACME") == "salary"
    assert rule_based("UPI-SWIGGY-PAYMENT-999") == "discretionary"
    assert rule_based("completely unknown string xyz") is None
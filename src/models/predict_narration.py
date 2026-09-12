"""
src/models/predict_narration.py
Phase 2 - Reusable inference helper for the narration classifier.

Usage:
    from src.models.predict_narration import predict_categories
    cats = predict_categories(["NEFT/XYZ/SAL/ACME", "UPI-SWIGGY-PAYMENT-123"])
"""

from pathlib import Path
import re
import joblib
import numpy as np

MODEL_PATH = Path("models/narration_clf.pkl")

_bundle = None  # lazy-loaded singleton


def _load():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Run: py -3.11 -m src.models.narration_clf"
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


# ---------------------------------------------------------------
# Rule-based fallback (used when model confidence is low)
# ---------------------------------------------------------------
RULES = [
    ("salary",           [r"\bsal\b", r"salary", r"sal/"]),
    ("emi",              [r"\bemi\b", r"loan\d", r"ecs-emi"]),
    ("rent",             [r"rent", r"landlord"]),
    ("bounce",           [r"insufficient", r"return", r"bounce"]),
    ("business_income",  [r"business", r"receipt", r"gst"]),
    ("suspicious",       [r"transfer-in", r"window", r"round.?trip"]),
    ("discretionary",    [r"swiggy", r"zomato", r"bigbasket", r"amazon",
                          r"flipkart", r"payment"]),
]


def rule_based(narration: str):
    """Return a category from rules, or None if no rule matches."""
    s = str(narration).lower()
    for label, patterns in RULES:
        for p in patterns:
            if re.search(p, s):
                return label
    return None


# ---------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------
def predict_categories(narrations, confidence_threshold: float = 0.6):
    """
    Predict categories for a list of narration strings.

    If model confidence < confidence_threshold, use the rule-based fallback.
    If fallback also fails, default to 'discretionary' (most common class).

    Returns:
        preds (np.ndarray)  - final predicted labels
        confs (np.ndarray)  - model confidence for each prediction
        used_rule (np.ndarray of bool) - whether the fallback was used
    """
    bundle = _load()
    kind = bundle["kind"]

    narrations = [str(n) for n in narrations]
    n = len(narrations)

    # ---- model probabilities ----
    if kind == "logreg":
        probs = bundle["model"].predict_proba(narrations)
        classes = bundle["model"].classes_
    else:
        X = bundle["vectorizer"].transform(narrations)
        probs = bundle["model"].predict_proba(X)
        classes = bundle["model"].classes_

    idx = np.argmax(probs, axis=1)
    model_pred = classes[idx]
    model_conf = probs[np.arange(n), idx]

    # ---- apply fallback where confidence is low ----
    final = model_pred.copy()
    used_rule = np.zeros(n, dtype=bool)
    for i in range(n):
        if model_conf[i] < confidence_threshold:
            r = rule_based(narrations[i])
            if r is not None:
                final[i] = r
                used_rule[i] = True
    return final, model_conf, used_rule


def predict_one(narration: str, confidence_threshold: float = 0.6) -> str:
    """Convenience wrapper for a single narration."""
    preds, _, _ = predict_categories([narration], confidence_threshold)
    return preds[0]


# ---------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "NEFT/XYZ/SAL/ACME-CORP",
        "ECS-EMI-BANK-HOMELOAN-2025",
        "UPI-SWIGGY-PAYMENT-4512",
        "ECS-RETURN-INSUFFICIENT FUNDS",
        "NEFT-TRANSFER-IN-300625",
        "UPI-BIGBASKET-PAYMENT-93123",
        "NEFT-RENT-LANDLORD-JAN2024",
    ]
    preds, confs, used = predict_categories(tests)
    print(f"{'narration':<40} {'pred':<18} {'conf':>6} {'rule?':>6}")
    print("-" * 75)
    for t, p, c, u in zip(tests, preds, confs, used):
        print(f"{t:<40} {p:<18} {c:>6.3f} {str(bool(u)):>6}")
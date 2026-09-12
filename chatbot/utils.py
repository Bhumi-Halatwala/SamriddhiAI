"""
Data + model helpers for the prototype.

IMPORTANT: generate_mock_customers() below is a STAND-IN so the app runs
end-to-end immediately. Replace load_customers() with a real pandas.read_csv()
of the Kaggle dataset once you've confirmed the actual column names, and swap
the rule-based scoring functions for your trained XGBoost/IsolationForest
models when they're ready.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

SEGMENTS = ["Young first-jobber", "Salaried saver", "Self-employed, high EMI", "Stable family, under-leveraged"]
PRODUCTS = ["Personal loan top-up", "Term insurance", "Recurring deposit / SIP", "Credit card upgrade"]
RISK_TIERS = ["Healthy", "Watch", "Early stress", "High stress"]


def load_customers(csv_path: str | None = None) -> pd.DataFrame:
    """Load real data if a path is given, else fall back to mock data."""
    if csv_path:
        return pd.read_csv(csv_path)
    return generate_mock_customers()


def generate_mock_customers(n: int = 40) -> pd.DataFrame:
    df = pd.DataFrame({
        "customer_id": [f"C{1000+i}" for i in range(n)],
        "name": [f"Customer {i+1}" for i in range(n)],
        "segment": np.random.choice(SEGMENTS, n),
        "monthly_income": np.random.randint(18000, 120000, n),
        "emi_to_income": np.round(np.random.uniform(0.05, 0.65, n), 2),
        "savings_rate": np.round(np.random.uniform(0.0, 0.35, n), 2),
        "missed_payments_3m": np.random.choice([0, 0, 0, 1, 2, 3], n),
        "recommended_product": np.random.choice(PRODUCTS, n),
    })
    df["risk_tier"] = df.apply(_risk_tier, axis=1)
    df["reason"] = df.apply(_reason_text, axis=1)
    return df


def _risk_tier(row) -> str:
    if row["missed_payments_3m"] >= 2 or row["emi_to_income"] > 0.55:
        return "High stress"
    if row["missed_payments_3m"] == 1 or row["emi_to_income"] > 0.4:
        return "Early stress"
    if row["emi_to_income"] > 0.3:
        return "Watch"
    return "Healthy"


def _reason_text(row) -> str:
    """Plain-language explanation - stand-in for SHAP output."""
    if row["recommended_product"] == "Personal loan top-up":
        return f"EMI-to-income is {row['emi_to_income']:.0%}, with room for a top-up at similar EMI."
    if row["recommended_product"] == "Term insurance":
        return f"Segment '{row['segment']}' with no active protection product on file."
    if row["recommended_product"] == "Recurring deposit / SIP":
        return f"Savings rate of {row['savings_rate']:.0%} with no active investment product."
    return "Consistent repayment history and available credit headroom."


def risk_badge_class(tier: str) -> str:
    return {
        "Healthy": "badge-green",
        "Watch": "badge-blue",
        "Early stress": "badge-amber",
        "High stress": "badge-red",
    }.get(tier, "badge-blue")


def intervention_for_tier(tier: str) -> str:
    return {
        "Healthy": "No action needed.",
        "Watch": "Send a gentle budgeting tip in-app.",
        "Early stress": "Proactive outreach offering EMI restructuring, before any payment is missed.",
        "High stress": "Route to a human relationship manager - no automated penalty action.",
    }.get(tier, "")

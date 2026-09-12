"""
src/chatbot/policy.py
Shared financial-stress policy for SamriddhiAI.
Single source of truth for the rule: high-stress customers must not be
offered new loans.
"""


def loan_policy(stress_level, is_loan_target=True):
    """Decide whether a customer can be offered new loans."""
    stress_level = (stress_level or "low").lower()

    if stress_level == "high":
        return {
            "allow": False,
            "reason_code": "STRESSED_HIGH",
            "reason_text": (
                "The customer is showing clear signs of financial stress. "
                "New loans must not be offered at this time."
            ),
            "alternative_actions": [
                "Arrange a check-in call - not a sales call",
                "Offer EMI restructuring or a repayment review",
                "Offer a free financial health consultation",
            ],
        }

    if not is_loan_target:
        return {
            "allow": False,
            "reason_code": "NOT_LOAN_TARGET",
            "reason_text": (
                "The customer is not currently eligible for new loan offers "
                "based on their life-stage segment."
            ),
            "alternative_actions": [
                "Review their profile in the next quarterly cycle",
                "Offer savings or insurance guidance instead",
            ],
        }

    if stress_level == "medium":
        return {
            "allow": True,
            "reason_code": "STRESSED_MEDIUM",
            "reason_text": (
                "The customer shows mild pressure signals. Offers may proceed, "
                "but frame them with care and mention repayment flexibility."
            ),
            "alternative_actions": [],
        }

    return {
        "allow": True,
        "reason_code": "OK",
        "reason_text": "No stress signals. Offers may proceed normally.",
        "alternative_actions": [],
    }

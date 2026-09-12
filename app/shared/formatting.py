"""app/shared/formatting.py
Plain-English helpers for bank staff UI.
"""

def stress_label(level):
    return {
        "low":    ("Calm",      "Customer finances look stable."),
        "medium": ("Watch",     "Some signs of financial pressure. Handle gently."),
        "high":   ("Concern",   "Strong signs of financial stress. Avoid product pushes."),
    }.get(level, ("Unknown", ""))


def alert_label(level):
    return {
        "low":    ("Normal",      "Behaviour is consistent with history."),
        "medium": ("Unusual",     "Some months look different from usual."),
        "high":   ("Investigate", "Patterns suggest a closer look is warranted."),
    }.get(level, ("Unknown", ""))


def money(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    if abs(x) >= 1e7:
        return f"Rs {x/1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"Rs {x/1e5:.2f} L"
    return f"Rs {x:,.0f}"


def pct(x, digits=1):
    try:
        return f"{float(x)*100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def explain_flag(flag_key):
    return {
        "hard_bounce":       "Missed payments (bounced transactions)",
        "hard_income_gap":   "Gap in monthly salary credits",
        "hard_savings_drop": "Savings dropped sharply for 2+ months",
        "hard_wd":           "Possible round-trip fund transfers at year-end",
        "hard_stress":       "Combined financial stress signals",
    }.get(flag_key, flag_key)


def explain_interest(product):
    return {
        "personal": "personal loans",
        "home":     "home loans",
        "auto":     "auto loans",
        "mortgage": "mortgage products",
    }.get(product, product)


def tier_name(row):
    if row.get("tier_metro", 0): return "Metro city"
    if row.get("tier_2", 0):     return "Tier 2 town"
    if row.get("tier_3", 0):     return "Tier 3 town"
    return "Unknown"


def occupation_name(row):
    if row.get("occ_salaried", 0): return "Salaried"
    if row.get("occ_self_emp", 0): return "Self-employed"
    if row.get("occ_business", 0): return "Business owner"
    return "Unknown"


def life_stage_name(stage):
    return {
        "young_renter":         "Young renter",
        "young_owner_no_debt":  "Young homeowner, no loans",
        "young_family":         "Young family",
        "mortgage_holder":      "Home loan customer",
        "established_borrower": "Existing borrower",
        "established_saver":    "Established saver",
        "pre_retiree":          "Near retirement",
        "debt_stressed":        "Financially stretched",
        "unclassified":         "Uncategorised",
    }.get(stage, stage)

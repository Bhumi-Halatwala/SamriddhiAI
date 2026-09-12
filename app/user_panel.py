"""app/user_panel.py
Customer-facing app for SamriddhiAI.

Run from SamriddhiAI/ root:
    py -3.11 -m streamlit run app/user_panel.py --server.port 8502
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path.cwd()))

from app.shared.data_loader import (
    load_model_input, load_stress, load_life_stage,
    load_txn_snapshot, load_recommender, get_row,
)
from app.shared.formatting import money, pct, explain_interest, life_stage_name

st.set_page_config(
    page_title="SamriddhiAI - My Banking",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded",
)


PRODUCT_NAMES = {
    "personal": "Personal Loan",
    "home":     "Home Loan",
    "auto":     "Car Loan",
    "mortgage": "Mortgage Top-Up",
}

PRODUCT_WHY = {
    "personal": "You can use this for weddings, education, medical needs, or any personal goal.",
    "home":     "If you are planning to buy a house, this loan can spread the cost over many years.",
    "auto":     "For buying a new car or two-wheeler, with predictable monthly payments.",
    "mortgage": "If you already own a home, this can help you unlock some of its value.",
}


def render_sidebar():
    st.sidebar.title("💚 SamriddhiAI")
    st.sidebar.caption("Your personal banking assistant")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sign in**")

    quick = st.sidebar.radio(
        "Quick demo login",
        ["Safe customer", "Stressed customer", "Enter my own ID"],
        key="user_panel_login_radio",
    )

    if quick == "Safe customer":
        customer_id = "CUST000005"
        st.sidebar.caption("Demo: a customer with healthy finances.")
    elif quick == "Stressed customer":
        customer_id = "CUST000001"
        st.sidebar.caption("Demo: a customer facing financial pressure.")
    else:
        customer_id = st.sidebar.text_input("Customer ID", "CUST000005")

    st.sidebar.markdown("---")
    st.sidebar.caption("Prototype · HackOut 2026 · Team Avengers")
    return customer_id


def render_my_account(row, life_row, stress_row):
    st.header("👤 My Account")
    st.caption("A quick look at your banking profile.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly income", money(row["monthly_declared_income"]))
    c2.metric("Monthly savings rate", pct(row["savings_rate_mean_7"]))
    c3.metric("EMI burden", pct(row["emi_to_income_mean_7"]))

    c4, c5, c6 = st.columns(3)
    c4.metric("Existing loans", int(row["existing_loan_count"]))
    c5.metric("Years with us", f"{row['relationship_tenure_years']:.1f}")
    c6.metric("Age", int(row["age"]))

    st.markdown("---")
    st.subheader("Where you are in life")
    if life_row is not None:
        st.info(f"**{life_stage_name(life_row['life_stage'])}**")

    if stress_row is not None and stress_row["stress_level"] == "high":
        st.warning(
            "💬 We noticed some months have been tighter than usual. "
            "If you would like to talk to someone about repayment options, "
            "we are here to help. No pressure, no pitches."
        )


def render_my_money(customer_id, row):
    st.header("💸 My Money")
    st.caption("Understand where your money goes each month.")

    monthly = pd.read_parquet("data/processed/txn_monthly.parquet")
    cust_monthly = monthly[monthly["customer_id"] == customer_id].copy()
    cust_monthly["month"] = pd.to_datetime(cust_monthly["month"])
    cust_monthly = cust_monthly.sort_values("month")

    if len(cust_monthly) == 0:
        st.info("No transaction history available.")
        return

    st.subheader("Money in vs money out (monthly)")
    chart_data = cust_monthly[["month", "credit_sum", "debit_sum"]].rename(
        columns={"month": "Month", "credit_sum": "Money in", "debit_sum": "Money out"}
    ).set_index("Month")
    st.line_chart(chart_data, height=260)

    st.subheader("Where your money goes (average per month)")
    cat_cols = {
        "emi_sum": "EMI",
        "rent_sum": "Rent",
        "discretionary_sum": "Everyday spending",
        "business_income_sum": "Business expenses",
    }
    present = {k: v for k, v in cat_cols.items() if k in cust_monthly.columns}
    cat_avgs = {label: float(cust_monthly[col].mean()) for col, label in present.items()}
    cat_avgs["Savings"] = max(0.0, float(cust_monthly["net_flow"].mean()))

    if sum(cat_avgs.values()) > 0:
        cat_df = pd.DataFrame({
            "Category": list(cat_avgs.keys()),
            "Average": list(cat_avgs.values()),
        }).set_index("Category")
        st.bar_chart(cat_df, height=260)

    st.subheader("Your savings trend")
    if "savings_rate" in cust_monthly.columns:
        sav = cust_monthly[["month", "savings_rate"]].rename(
            columns={"month": "Month", "savings_rate": "Savings rate"}
        ).set_index("Month")
        st.line_chart(sav, height=220)

    st.markdown("---")
    st.subheader("What we noticed")
    observations = []

    avg_savings = float(cust_monthly["savings_rate"].mean())
    if avg_savings > 0.3:
        observations.append("You save more than 30% of your income on average - excellent.")
    elif avg_savings > 0.1:
        observations.append("You save a steady amount every month. Keep it up.")
    elif avg_savings >= 0:
        observations.append("Your savings are thin. Try setting aside a small amount each month.")
    else:
        observations.append("Some months, you spend more than you earn. We can help you plan.")

    for prod_key, label in [("home", "home"), ("auto", "car"), ("personal", "personal")]:
        if row.get(f"interest_{prod_key}", 0) > 0:
            observations.append(f"You recently checked out {label} loans.")

    emi_burden = float(row["emi_to_income_mean_7"])
    if emi_burden > 0.4:
        observations.append("Your EMIs take up a large share of your income - over 40%.")
    elif emi_burden > 0.2:
        observations.append("Your EMIs are a moderate share of your income.")

    if not observations:
        observations.append("Your finances look stable. Nothing urgent stands out.")

    for obs in observations:
        st.markdown(f"- {obs}")


def render_for_you(bundle, row, stress_row):
    st.header("✨ For You")
    st.caption("Personalized suggestions based on your profile and activity.")

    if stress_row is not None and stress_row["stress_level"] == "high":
        st.info(
            "We want to help you first. Before we talk about new products, "
            "let us look at your current commitments together."
        )
        st.markdown("### Things that might help right now")
        st.markdown("- **Talk to a financial advisor** - free, no obligation")
        st.markdown("- **Restructure your EMIs** - we can spread payments over a longer time")
        st.markdown("- **Savings plan review** - a small buffer goes a long way")
        return

    feature_names = bundle["feature_names"]
    x = row[feature_names].to_frame().T.astype(float).fillna(0)
    p_apply = float(bundle["head_a"].predict_proba(x)[0, 1])
    p_product = bundle["head_b"].predict_proba(x)[0]
    products = bundle["head_b"].classes_
    scores = sorted(
        [(p, p_apply * pr) for p, pr in zip(products, p_product)],
        key=lambda t: -t[1],
    )
    top2 = scores[:2]

    st.success(f"**We found {len(top2)} products that may suit you.**")

    for i, (prod, score) in enumerate(top2, 1):
        name = PRODUCT_NAMES.get(prod, prod.title())
        st.markdown(f"### {i}. {name}")
        st.markdown(PRODUCT_WHY.get(prod, ""))
        if i == 1:
            st.caption("This one seems like the best match for you right now.")
        st.markdown("---")

    st.subheader("Why we think this")
    reasons = []
    if row.get("savings_rate_mean_7", 0) > 0.3:
        reasons.append("You save regularly, so repayments would fit comfortably.")
    if row.get("salary_months_present", 0) >= 12:
        reasons.append("You receive a steady salary each month.")
    if row.get("has_any_existing_loan", 0) == 0:
        reasons.append("You have no existing loans, which keeps things simple.")
    for prod_key in PRODUCT_NAMES:
        if row.get(f"interest_{prod_key}", 0) > 0:
            reasons.append(f"You recently viewed {explain_interest(prod_key)}.")
    if not reasons:
        reasons.append("This is a general suggestion based on your profile.")

    for r in reasons[:5]:
        st.markdown(f"- {r}")

    st.caption(
        "These suggestions are meant to be helpful, not pushy. "
        "You are always free to say no."
    )


def main():
    model_input = load_model_input()
    stress = load_stress()
    life_stage = load_life_stage()
    bundle = load_recommender()

    customer_id = render_sidebar()

    row = get_row(model_input, customer_id)
    if row is None:
        st.error(f"Customer {customer_id} not found.")
        return

    stress_row = get_row(stress, customer_id)
    life_row = get_row(life_stage, customer_id)

    st.title(f"Hi, {customer_id}")
    st.caption("Welcome back. Here is a quick look at your money.")

    tabs = st.tabs(["👤 My Account", "💸 My Money", "✨ For You"])

    with tabs[0]:
        render_my_account(row, life_row, stress_row)
    with tabs[1]:
        render_my_money(customer_id, row)
    with tabs[2]:
        render_for_you(bundle, row, stress_row)


if __name__ == "__main__":
    main()

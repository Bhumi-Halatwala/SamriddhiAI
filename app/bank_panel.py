"""app/bank_panel.py
Phase 7 - Bank admin panel for SamriddhiAI.

Audience: bank relationship managers / branch staff.
Purpose:  full customer picture + next-best-action + risk alerts.

Chatbot is NOT included here - it lives in the customer app (app/user_panel.py).

Run from SamriddhiAI/ root:
    py -3.11 -m streamlit run app/bank_panel.py
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path.cwd()))

from app.shared.data_loader import (
    load_model_input, load_stress, load_anomaly, load_life_stage,
    load_recommender, get_row,
)
from app.shared.formatting import (
    stress_label, alert_label, money, pct,
    explain_flag, explain_interest, tier_name, occupation_name, life_stage_name,
)

st.set_page_config(
    page_title="SamriddhiAI - Bank Panel",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_recommendation(bundle, customer_row, stress_row):
    if stress_row is not None and stress_row.get("stress_level") == "high":
        return {
            "suppressed": True,
            "reason": "This customer is showing signs of financial stress. "
                      "Product offers should be paused and handled with care.",
            "alternatives": [
                "Call to check in - not to sell",
                "Offer EMI restructuring or a repayment plan review",
                "Offer a free financial health consultation",
            ],
        }

    feature_names = bundle["feature_names"]
    x = customer_row[feature_names].to_frame().T.astype(float).fillna(0)
    p_apply = float(bundle["head_a"].predict_proba(x)[0, 1])
    p_product = bundle["head_b"].predict_proba(x)[0]
    products = bundle["head_b"].classes_

    scores = sorted(
        [(p, p_apply * pr) for p, pr in zip(products, p_product)],
        key=lambda t: -t[1],
    )
    return {
        "suppressed": False,
        "apply_probability": p_apply,
        "top_products": [{"product": p, "score": float(s)} for p, s in scores[:2]],
    }


def reason_codes(customer_row):
    r = []
    if customer_row.get("engagement_score", 0) > 2.5:
        r.append("Uses the app often (logins, loan calculator)")
    for prod in ["home", "auto", "personal", "mortgage"]:
        if customer_row.get(f"interest_{prod}", 0) > 0:
            r.append(f"Recently viewed {explain_interest(prod)}")
    if customer_row.get("savings_rate_mean_7", 0) > 0.3:
        r.append("Saves over 30% of monthly income")
    if customer_row.get("salary_months_present", 0) >= 12:
        r.append("Receives regular monthly salary for 12+ months")
    if customer_row.get("has_any_existing_loan", 0) == 0:
        r.append("No existing loans on file")
    if customer_row.get("bureau_score_proxy", 0) >= 750:
        r.append(f"Strong credit score ({int(customer_row.get('bureau_score_proxy', 0))})")
    if not r:
        r.append("Standard profile - no distinguishing signals yet")
    return r[:4]


def render_sidebar(model_input, life_stage):
    st.sidebar.title("🏦 SamriddhiAI - Bank Panel")
    st.sidebar.caption("For branch staff and relationship managers")

    mode = st.sidebar.radio(
        "How do you want to find a customer?",
        ["Pick from a list", "Enter customer ID", "Filter by category"],
        key="bank_panel_mode_radio",
    )

    if mode == "Pick from a list":
        sample_ids = model_input["customer_id"].sample(30, random_state=42).tolist()
        customer_id = st.sidebar.selectbox("Customer", sample_ids)

    elif mode == "Enter customer ID":
        customer_id = st.sidebar.text_input("Customer ID", "CUST000005")

    else:
        segment = st.sidebar.selectbox(
            "Category",
            [
                "Financially stretched (handle with care)",
                "Young family",
                "Home loan customers",
                "Young renters",
                "Existing borrowers",
                "Established savers",
            ],
        )
        mapping = {
            "Financially stretched (handle with care)": "debt_stressed",
            "Young family": "young_family",
            "Home loan customers": "mortgage_holder",
            "Young renters": "young_renter",
            "Existing borrowers": "established_borrower",
            "Established savers": "established_saver",
        }
        seg_key = mapping[segment]
        ids = life_stage[life_stage["life_stage"] == seg_key]["customer_id"].head(30).tolist()
        customer_id = st.sidebar.selectbox("Customer", ids)

    st.sidebar.markdown("---")
    st.sidebar.caption("Prototype · HackOut 2026 · Team Avengers")
    return customer_id


def render_snapshot(row, life_row, stress_row, anomaly_row):
    st.header("👤 Customer snapshot")
    st.caption("Everything you would want to know before picking up the phone.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", int(row["age"]))
    c2.metric("Location", tier_name(row))
    c3.metric("Monthly income", money(row["monthly_declared_income"]))
    c4.metric("Credit score", int(row["bureau_score_proxy"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Existing loans", int(row["existing_loan_count"]))
    c6.metric("Saves per month", pct(row["savings_rate_mean_7"]))
    c7.metric("EMI burden", pct(row["emi_to_income_mean_7"]))
    c8.metric("Banking with us", f"{row['relationship_tenure_years']:.1f} yrs")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer category")
        if life_row is not None:
            st.success(f"**{life_stage_name(life_row['life_stage'])}**")
            if life_row["is_loan_target"]:
                st.caption("Safe to approach with product offers.")
            else:
                st.caption("Do NOT pitch products. Prioritise support.")
        st.subheader("Occupation")
        st.write(occupation_name(row))

    with col2:
        st.subheader("What they are interested in")
        interests = {
            "Personal loans": int(row.get("interest_personal", 0)),
            "Home loans": int(row.get("interest_home", 0)),
            "Auto loans": int(row.get("interest_auto", 0)),
            "Mortgage": int(row.get("interest_mortgage", 0)),
        }
        if sum(interests.values()) == 0:
            st.caption("No recent interest signals.")
        else:
            for name, n in interests.items():
                if n > 0:
                    st.write(f"- {name}: **{n}** views")

    st.markdown("---")
    st.subheader("Health check")
    h1, h2 = st.columns(2)
    with h1:
        if stress_row is not None:
            label, desc = stress_label(stress_row["stress_level"])
            color = {"Calm": "🟢", "Watch": "🟡", "Concern": "🔴"}[label]
            st.markdown(f"### {color} Financial health: **{label}**")
            st.caption(desc)
    with h2:
        if anomaly_row is not None:
            label, desc = alert_label(anomaly_row["alert_level"])
            color = {"Normal": "🟢", "Unusual": "🟡", "Investigate": "🔴"}[label]
            st.markdown(f"### {color} Behaviour: **{label}**")
            st.caption(desc)


def render_next_action(bundle, row, stress_row):
    st.header("🎯 Next best action")
    st.caption("What to offer - and what to avoid - based on this customer's profile.")

    rec = get_recommendation(bundle, row, stress_row)

    if rec["suppressed"]:
        st.error("🚫 Do not offer new loans to this customer.")
        st.markdown(f"**Why:** {rec['reason']}")
        st.markdown("### Suggested actions instead")
        for alt in rec["alternatives"]:
            st.markdown(f"- {alt}")
        return

    st.success(f"Likely to apply for a loan: {rec['apply_probability']:.0%}")
    st.caption("Model confidence based on behavioral and transaction signals.")

    st.markdown("### Recommended products")
    for i, item in enumerate(rec["top_products"], 1):
        st.markdown(f"**#{i} {item['product'].title()} loan** - match score {item['score']:.2f}")

    st.markdown("---")
    st.subheader("Why this recommendation?")
    for r in reason_codes(row):
        st.markdown(f"- {r}")

    st.caption("Reason codes are deterministic business rules for regulatory explainability.")


def render_alerts(row, stress_row, anomaly_row):
    st.header("🚨 Risk and alerts")
    st.caption("Early warning signals to prevent problems before they escalate.")

    if stress_row is None or anomaly_row is None:
        st.warning("No alert data available for this customer.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Financial stress")
        label, desc = stress_label(stress_row["stress_level"])
        color = {"Calm": "green", "Watch": "orange", "Concern": "red"}[label]
        st.markdown(f"### :{color}[{label}]")
        st.caption(desc)
        st.markdown("**What we see:**")
        st.markdown(f"- Missed payments: {pct(max(0, stress_row['z_bounce'])/5)}")
        st.markdown(f"- Savings trend: {pct(max(0, stress_row['z_savings'])/5)}")
        st.markdown(f"- Unusual spending: {pct(max(0, stress_row['z_discretionary'])/5)}")
        st.markdown(f"- Salary trend: {pct(max(0, stress_row['z_salary'])/5)}")
        st.caption("Higher percentages mean more concern.")

    with col2:
        st.subheader("Behavioural anomalies")
        label, desc = alert_label(anomaly_row["alert_level"])
        color = {"Normal": "green", "Unusual": "orange", "Investigate": "red"}[label]
        st.markdown(f"### :{color}[{label}]")
        st.caption(desc)

        st.markdown("**Detected patterns:**")
        flags = []
        for key in ["hard_bounce", "hard_income_gap", "hard_savings_drop", "hard_wd", "hard_stress"]:
            if anomaly_row.get(key, 0):
                flags.append(explain_flag(key))
        if flags:
            for f in flags:
                st.markdown(f"- {f}")
        else:
            st.caption("No specific patterns detected.")

    if stress_row["stress_level"] == "high" or anomaly_row["alert_level"] == "high":
        st.warning(
            "Suggested next step: Reach out with care. "
            "Frame the call as a check-in, not a sales pitch. "
            "Offer restructuring, EMI relief, or a financial health consultation."
        )


def main():
    model_input = load_model_input()
    stress = load_stress()
    anomaly = load_anomaly()
    life_stage = load_life_stage()
    bundle = load_recommender()

    customer_id = render_sidebar(model_input, life_stage)

    row = get_row(model_input, customer_id)
    if row is None:
        st.error(f"Customer {customer_id} not found.")
        return

    stress_row = get_row(stress, customer_id)
    anomaly_row = get_row(anomaly, customer_id)
    life_row = get_row(life_stage, customer_id)

    st.title(f"Customer: {customer_id}")

    tabs = st.tabs(["👤 Snapshot", "🎯 Next best action", "🚨 Risk and alerts"])

    with tabs[0]:
        render_snapshot(row, life_row, stress_row, anomaly_row)
    with tabs[1]:
        render_next_action(bundle, row, stress_row)
    with tabs[2]:
        render_alerts(row, stress_row, anomaly_row)


if __name__ == "__main__":
    main()

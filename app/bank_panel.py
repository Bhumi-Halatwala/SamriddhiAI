"""app/bank_panel.py
Bank admin panel for SamriddhiAI.

Run from SamriddhiAI/ root:
    py -3.11 -m streamlit run app/bank_panel.py --server.port 8501
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import altair as alt

sys.path.insert(0, str(Path.cwd()))

from app.shared.data_loader import (
    load_model_input, load_stress, load_anomaly, load_life_stage,
    load_recommender, get_row,
)
from app.shared.formatting import (
    stress_label, alert_label, money, pct,
    explain_flag, explain_interest, tier_name, occupation_name, life_stage_name,
)
from app.shared.style import (
    apply_brand_style, header, viewing_label, metric_card, status_pill,
    reco_card, alert_banner, hr,
)

st.set_page_config(
    page_title="SamriddhiAI - Bank Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_brand_style(mode="bank")


PRODUCT_DISPLAY = {
    "personal": "Personal Loan",
    "home":     "Home Loan",
    "auto":     "Auto Loan",
    "mortgage": "Mortgage Top-Up",
}

CONTACT_METHOD = {
    "application_intent": "Call the customer - they are actively exploring products.",
    "browsing":           "Send an in-app message - they are engaged and browsing.",
    "logged_in":          "Send an SMS - light touch, occasional app activity.",
    "inactive":           "Branch visit or phone call - low recent app activity.",
}


def get_recommendation(bundle, customer_row, stress_row):
    if stress_row is not None and stress_row.get("stress_level") == "high":
        return {
            "suppressed": True,
            "reason": "Signs of financial stress detected. "
                      "Product offers should be paused and routed to advisory.",
            "alternatives": [
                "Place a check-in call - not a sales call",
                "Offer EMI restructuring or a repayment review",
                "Offer a financial health consultation",
            ],
        }

    feature_names = bundle["feature_names"]
    x = customer_row[feature_names].to_frame().T.astype(float).fillna(0)
    p_apply = float(bundle["head_a"].predict_proba(x)[0, 1])
    p_product = bundle["head_b"].predict_proba(x)[0]
    products = bundle["head_b"].classes_

    scored = sorted(
        [(p, p_apply * pr) for p, pr in zip(products, p_product)],
        key=lambda t: -t[1],
    )
    scored = [(p, s) for p, s in scored if s > 0.001][:2]

    return {
        "suppressed": False,
        "apply_probability": p_apply,
        "top_products": [{"product": p, "score": float(s)} for p, s in scored],
    }


def reason_codes(customer_row):
    r = []
    if customer_row.get("engagement_score", 0) > 2.5:
        r.append("Active app user - frequent logins and calculator usage")
    for prod in ["home", "auto", "personal", "mortgage"]:
        if customer_row.get(f"interest_{prod}", 0) > 0:
            r.append(f"Recently viewed {explain_interest(prod)}")
    if customer_row.get("savings_rate_mean_7", 0) > 0.3:
        r.append("Savings rate above 30% of monthly income for 7 months")
    if customer_row.get("salary_months_present", 0) >= 12:
        r.append("Regular monthly salary credited for 12+ months")
    if customer_row.get("has_any_existing_loan", 0) == 0:
        r.append("No existing loans on file")
    if customer_row.get("bureau_score_proxy", 0) >= 750:
        r.append(f"Bureau score of {int(customer_row.get('bureau_score_proxy', 0))}")
    if not r:
        r.append("Standard profile - no distinguishing signals")
    return r[:5]


def portfolio_summary():
    stress = load_stress()
    anomaly = load_anomaly()
    total = len(stress)
    stressed = int((stress["stress_level"] == "high").sum())
    investigate = int((anomaly["alert_level"] == "high").sum())
    return total, stressed, investigate


def render_header():
    st.markdown(
        header(
            "SamriddhiAI",
            "Relationship banking for Bharat",
            "Bank Console",
        ),
        unsafe_allow_html=True,
    )


def render_sidebar(model_input, life_stage):
    st.sidebar.markdown("### Find a customer")

    mode = st.sidebar.radio(
        "Search mode",
        ["From a list", "By customer ID", "By segment"],
        key="bank_panel_mode_radio",
    )

    if mode == "From a list":
        sample_ids = model_input["customer_id"].sample(30, random_state=42).tolist()
        customer_id = st.sidebar.selectbox("Customer", sample_ids, key="bank_pick_list")

    elif mode == "By customer ID":
        customer_id = st.sidebar.text_input(
            "Customer ID",
            placeholder="e.g. CUST000005",
            key="bank_pick_id",
        )
        if not customer_id:
            customer_id = "CUST000005"

    else:
        segment = st.sidebar.selectbox(
            "Segment",
            [
                "Financially stretched",
                "Young family",
                "Home loan customers",
                "Young renters",
                "Existing borrowers",
                "Established savers",
            ],
            key="bank_pick_segment",
        )
        mapping = {
            "Financially stretched": "debt_stressed",
            "Young family": "young_family",
            "Home loan customers": "mortgage_holder",
            "Young renters": "young_renter",
            "Existing borrowers": "established_borrower",
            "Established savers": "established_saver",
        }
        seg_key = mapping[segment]
        ids = life_stage[life_stage["life_stage"] == seg_key]["customer_id"].head(30).tolist()
        customer_id = st.sidebar.selectbox("Customer", ids, key="bank_pick_seg_cust")

    st.sidebar.markdown("---")
    total, stressed, investigate = portfolio_summary()
    st.sidebar.markdown("### Portfolio at a glance")
    st.sidebar.markdown(f"- Customers: **{total:,}**")
    st.sidebar.markdown(f"- High stress: **{stressed:,}**")
    st.sidebar.markdown(f"- Under review: **{investigate:,}**")

    st.sidebar.markdown("---")
    st.sidebar.caption("SamriddhiAI · HackOut 2026 · Team Avengers")
    return customer_id


# ------------------------------------------------------------------
# Tab 1 — Snapshot
# ------------------------------------------------------------------
def render_snapshot(row, life_row, stress_row, anomaly_row, customer_id):
    st.subheader("Customer snapshot")

    st.markdown("##### Customer profile")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Age", str(int(row["age"]))), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("City type", tier_name(row)), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Occupation", occupation_name(row)), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Banking tenure",
                                f"{row['relationship_tenure_years']:.1f} yrs",
                                "years as our customer"), unsafe_allow_html=True)

    st.markdown("##### Financial position")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Monthly income",
                                money(row["monthly_declared_income"])), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Savings rate",
                                pct(row["savings_rate_mean_7"]),
                                "share of monthly income"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("EMI payments",
                                pct(row["emi_to_income_mean_7"]),
                                "share of monthly income"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Existing loans",
                                str(int(row["existing_loan_count"]))), unsafe_allow_html=True)

    st.markdown("##### Credit and status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Bureau score",
                                str(int(row["bureau_score_proxy"]))), unsafe_allow_html=True)
    with c2:
        if stress_row is not None:
            level = stress_row["stress_level"]
            pill_level = {"low": "ok", "medium": "watch", "high": "concern"}[level]
            label, _ = stress_label(level)
            st.markdown(metric_card("Financial health", label), unsafe_allow_html=True)
            st.markdown(status_pill(pill_level), unsafe_allow_html=True)
    with c3:
        if anomaly_row is not None:
            level = anomaly_row["alert_level"]
            pill_level = {"low": "ok", "medium": "watch", "high": "concern"}[level]
            label, _ = alert_label(level)
            st.markdown(metric_card("Behaviour", label), unsafe_allow_html=True)
            st.markdown(status_pill(pill_level), unsafe_allow_html=True)
    with c4:
        if life_row is not None:
            st.markdown(metric_card("Segment",
                                    life_stage_name(life_row["life_stage"])),
                        unsafe_allow_html=True)

    st.markdown(hr(), unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("##### Interest signals")
        interests = {
            "Personal loans": int(row.get("interest_personal", 0)),
            "Home loans":     int(row.get("interest_home", 0)),
            "Auto loans":     int(row.get("interest_auto", 0)),
            "Mortgage":       int(row.get("interest_mortgage", 0)),
        }
        active = {k: v for k, v in interests.items() if v > 0}
        if not active:
            st.caption("No recent interest signals recorded.")
        else:
            for name, n in active.items():
                st.markdown(f"- {name}: **{n}** view{'s' if n > 1 else ''}")

    with col2:
        st.markdown("##### Eligibility")
        if life_row is not None and life_row["is_loan_target"]:
            st.markdown("**Eligible for product offers.** "
                        "Customer profile and stress level permit recommendations.")
        else:
            st.markdown("**Not eligible for product offers.** "
                        "Route to advisory or support.")

    st.markdown(hr(), unsafe_allow_html=True)

    st.markdown("##### Recent activity (last 6 months)")
    monthly = pd.read_parquet("data/processed/txn_monthly.parquet")
    cust_monthly = monthly[monthly["customer_id"] == customer_id].copy()
    cust_monthly["month"] = pd.to_datetime(cust_monthly["month"])
    cust_monthly = cust_monthly.sort_values("month").tail(6)

    if len(cust_monthly) > 0:
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.markdown(metric_card("Avg monthly credits",
                                    money(cust_monthly["credit_sum"].mean())),
                        unsafe_allow_html=True)
        with a2:
            st.markdown(metric_card("Avg monthly debits",
                                    money(cust_monthly["debit_sum"].mean())),
                        unsafe_allow_html=True)
        with a3:
            st.markdown(metric_card("Bounces (6m)",
                                    str(int(cust_monthly["bounce_count"].sum()))),
                        unsafe_allow_html=True)
        with a4:
            days_since = int(row.get("login_days_since_last", 0))
            st.markdown(metric_card("Days since last login",
                                    str(days_since) if days_since else "0"),
                        unsafe_allow_html=True)
    else:
        st.caption("No recent transaction data.")


# ------------------------------------------------------------------
# Tab 2 — Next best action
# ------------------------------------------------------------------
def render_next_action(bundle, row, stress_row):
    st.subheader("Next best action")
    st.caption("Recommended approach for the next customer interaction.")

    rec = get_recommendation(bundle, row, stress_row)

    if rec["suppressed"]:
        st.markdown(
            alert_banner(
                "Do not offer new loans to this customer",
                rec["reason"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown("##### Suggested actions")
        for alt in rec["alternatives"]:
            st.markdown(f"- {alt}")
        
        return

    apply_pct = rec["apply_probability"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            metric_card("Likelihood to apply",
                        f"{apply_pct:.0%}",
                        "model estimate based on transaction and behavioural signals"),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("##### Recommended contact method")
        funnel = row.get("funnel_application_intent", 0)
        if funnel == 1:
            method = CONTACT_METHOD["application_intent"]
        elif row.get("funnel_browsing", 0) == 1:
            method = CONTACT_METHOD["browsing"]
        elif row.get("funnel_logged_in", 0) == 1:
            method = CONTACT_METHOD["logged_in"]
        else:
            method = CONTACT_METHOD["inactive"]
        st.markdown(method)

    if not rec["top_products"]:
        st.info("No product currently shows a strong match. "
                "Consider a general relationship review call.")
    else:
        st.markdown("##### Recommended products")
        for i, item in enumerate(rec["top_products"], 1):
            product_name = PRODUCT_DISPLAY.get(item["product"], item["product"].title())
            body = (
                f"Match score {item['score']:.2f}. "
                "Ranked by combining likelihood to apply with product fit."
            )
            st.markdown(
                reco_card(f"{i}. {product_name}", body),
                unsafe_allow_html=True,
            )

    st.markdown(hr(), unsafe_allow_html=True)
    st.markdown("##### Why this recommendation")
    for r in reason_codes(row):
        st.markdown(f"- {r}")
    st.caption(
        "Reason codes are deterministic business rules. They are shown so that "
        "every recommendation can be explained to the customer or to an auditor."
    )


# ------------------------------------------------------------------
# Tab 3 — Risk and alerts
# ------------------------------------------------------------------
def render_alerts(row, stress_row, anomaly_row):
    st.subheader("Risk and alerts")
    st.caption("Early warning signals to help prevent escalation.")

    if stress_row is None or anomaly_row is None:
        st.info("No alert data for this customer.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Financial health")
        label, desc = stress_label(stress_row["stress_level"])
        level = stress_row["stress_level"]
        pill_level = {"low": "ok", "medium": "watch", "high": "concern"}[level]
        st.markdown(f"**{label}** {status_pill(pill_level)}", unsafe_allow_html=True)
        st.caption(desc)
        st.markdown("Signals relative to this customer's own history:")
        for label_txt, val in [
            ("Missed payments",  stress_row["z_bounce"]),
            ("Savings trend",    stress_row["z_savings"]),
            ("Unusual spending", stress_row["z_discretionary"]),
            ("Salary trend",     stress_row["z_salary"]),
        ]:
            intensity = min(1.0, max(0.0, float(val) / 5.0))
            st.markdown(f"- {label_txt}: {intensity:.0%}")

    with col2:
        st.markdown("##### Behavioural anomalies")
        label, desc = alert_label(anomaly_row["alert_level"])
        level = anomaly_row["alert_level"]
        pill_level = {"low": "ok", "medium": "watch", "high": "concern"}[level]
        st.markdown(f"**{label}** {status_pill(pill_level)}", unsafe_allow_html=True)
        st.caption(desc)

        flags = []
        for key in ["hard_bounce", "hard_income_gap", "hard_savings_drop", "hard_wd", "hard_stress"]:
            if anomaly_row.get(key, 0):
                flags.append(explain_flag(key))

        if flags:
            st.markdown("Detected patterns:")
            for f in flags:
                st.markdown(f"- {f}")
        else:
            st.caption("No specific anomalies detected in the observation window.")

    if stress_row["stress_level"] == "high" or anomaly_row["alert_level"] == "high":
        st.markdown(hr(), unsafe_allow_html=True)
        st.markdown(
            alert_banner(
                "Suggested approach",
                "Reach out with care. Frame the call as a check-in, "
                "not a sales pitch. Offer restructuring, EMI relief, or a "
                "financial health consultation.",
            ),
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    model_input = load_model_input()
    stress = load_stress()
    anomaly = load_anomaly()
    life_stage = load_life_stage()
    bundle = load_recommender()

    render_header()
    customer_id = render_sidebar(model_input, life_stage)

    row = get_row(model_input, customer_id)
    if row is None:
        st.error(f"Customer {customer_id} not found.")
        return

    stress_row = get_row(stress, customer_id)
    anomaly_row = get_row(anomaly, customer_id)
    life_row = get_row(life_stage, customer_id)

    st.markdown(viewing_label(customer_id), unsafe_allow_html=True)

    tabs = st.tabs(["Snapshot", "Next best action", "Risk and alerts"])

    with tabs[0]:
        render_snapshot(row, life_row, stress_row, anomaly_row, customer_id)
    with tabs[1]:
        render_next_action(bundle, row, stress_row)
    with tabs[2]:
        render_alerts(row, stress_row, anomaly_row)


if __name__ == "__main__":
    main()

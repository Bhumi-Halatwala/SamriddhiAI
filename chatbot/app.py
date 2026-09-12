import streamlit as st
import pandas as pd
from utils import load_customers, risk_badge_class, intervention_for_tier

st.set_page_config(page_title="Bharat Banking AI", page_icon="", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>Bharat Banking AI</h1>
        <span>Hyper-personalized banking prototype</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Load data (swap None for a real CSV path once columns are confirmed) ----
df = load_customers(csv_path=None)

tab_dashboard, tab_risk, tab_chat = st.tabs(["Customer dashboard", "Risk monitor", "Onboarding chatbot"])

# ============================================================
# TAB 1 - Personalized recommendation dashboard
# ============================================================
with tab_dashboard:
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="card"><h3>Select a customer</h3></div>', unsafe_allow_html=True)
        customer_id = st.selectbox("", df["customer_id"], label_visibility="collapsed")
        row = df[df["customer_id"] == customer_id].iloc[0]

        m1, m2, m3 = st.columns(3)
        for col, (label, value) in zip(
            (m1, m2, m3),
            [
                ("Monthly income", f"₹{row['monthly_income']:,}"),
                ("EMI / income", f"{row['emi_to_income']:.0%}"),
                ("Savings rate", f"{row['savings_rate']:.0%}"),
            ],
        ):
            col.markdown(
                f'<div class="metric-tile"><div class="value">{value}</div>'
                f'<div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(
            f"""
            <div class="card">
                <h3>Segment</h3>
                <span class="badge badge-purple">{row['segment']}</span>
            </div>
            <div class="card">
                <h3>Recommended product</h3>
                <span class="badge badge-blue">{row['recommended_product']}</span>
                <p style="margin-top:0.6rem;">{row['reason']}</p>
            </div>
            <div class="card">
                <h3>Risk tier</h3>
                <span class="badge {risk_badge_class(row['risk_tier'])}">{row['risk_tier']}</span>
                <p style="margin-top:0.6rem;">{intervention_for_tier(row['risk_tier'])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# TAB 2 - Portfolio-level risk monitor
# ============================================================
with tab_risk:
    st.markdown('<div class="card"><h3>Risk distribution across customers</h3></div>', unsafe_allow_html=True)
    counts = df["risk_tier"].value_counts().reindex(["Healthy", "Watch", "Early stress", "High stress"]).fillna(0)
    st.bar_chart(counts)

    st.markdown('<div class="card"><h3>Customers needing attention</h3></div>', unsafe_allow_html=True)
    flagged = df[df["risk_tier"].isin(["Early stress", "High stress"])][
        ["customer_id", "name", "segment", "risk_tier", "emi_to_income", "missed_payments_3m"]
    ]
    st.dataframe(flagged, use_container_width=True, hide_index=True)

# ============================================================
# TAB 3 - Mock vernacular onboarding chatbot
# ============================================================
with tab_chat:
    st.markdown('<div class="card"><h3>Onboarding journey demo (Hindi + English)</h3></div>', unsafe_allow_html=True)

    lang = st.radio("Language / भाषा", ["English", "हिंदी"], horizontal=True)

    script_en = [
        ("bot", "Welcome! Let's get your loan application started. What is your monthly income?"),
        ("user", "About ₹35,000 per month"),
        ("bot", "Got it. What is this loan for?"),
        ("user", "Buying a two-wheeler"),
        ("bot", "Thanks. I'll need your consent to check your credit bureau data - is that okay?"),
        ("user", "Yes, I agree"),
        ("bot", "You're pre-approved for a two-wheeler loan up to ₹1,20,000. A representative will confirm details in your language."),
    ]
    script_hi = [
        ("bot", "स्वागत है! चलिए आपका लोन आवेदन शुरू करते हैं। आपकी मासिक आय कितनी है?"),
        ("user", "लगभग ₹35,000 प्रति माह"),
        ("bot", "ठीक है। यह लोन किसके लिए है?"),
        ("user", "टू-व्हीलर खरीदने के लिए"),
        ("bot", "धन्यवाद। क्रेडिट ब्यूरो डेटा जांचने के लिए आपकी सहमति चाहिए - क्या यह ठीक है?"),
        ("user", "हाँ, मैं सहमत हूँ"),
        ("bot", "आप ₹1,20,000 तक के टू-व्हीलर लोन के लिए पूर्व-स्वीकृत हैं। एक प्रतिनिधि आपकी भाषा में विवरण की पुष्टि करेगा।"),
    ]

    script = script_hi if lang == "हिंदी" else script_en
    for speaker, text in script:
        css_class = "chat-bot" if speaker == "bot" else "chat-user"
        st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)

    st.caption("This is a scripted demo of the guided-journey pattern - production version would use STT/TTS + an LLM for free-text clarification within each fixed step.")

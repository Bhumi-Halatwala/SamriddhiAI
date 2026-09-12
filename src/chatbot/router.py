"""
src/chatbot/router.py
Phase 6 - Chatbot response engine.

Holds conversation state per session, detects intent, applies guardrails,
and produces a personalized reply using the customer's data.

Guardrail: if the customer's stress_level is 'high', refuse new-loan offers
and redirect to advisory.
"""

from pathlib import Path
import pandas as pd
from src.chatbot.languages import LANGUAGES, t
from src.chatbot.intents import detect_intent

PROCESSED = Path("data/processed")

_customer_data = None  # lazy-loaded


def _load_customer_data():
    global _customer_data
    if _customer_data is None:
        stress = pd.read_parquet(PROCESSED / "stress_features.parquet")
        profile = pd.read_parquet(PROCESSED / "profile_features.parquet")
        txn = pd.read_parquet(PROCESSED / "txn_snapshot.parquet")
        behavior = pd.read_parquet(PROCESSED / "behavior_features.parquet")

        df = profile[["customer_id", "age", "monthly_declared_income",
                      "existing_loan_count"]].copy()
        df = df.merge(stress[["customer_id", "stress_level", "is_stressed",
                              "stress_score"]],
                      on="customer_id", how="left")
        df = df.merge(txn[["customer_id", "savings_rate_mean_7",
                           "emi_to_income_mean_7", "emi_sum_mean_7",
                           "upi_share_mean_7"]],
                      on="customer_id", how="left")
        df = df.merge(behavior[["customer_id", "funnel_application_intent",
                                "interest_personal", "interest_home",
                                "interest_auto", "interest_mortgage"]],
                      on="customer_id", how="left")
        _customer_data = df.set_index("customer_id")

    return _customer_data


class ChatSession:
    """A single chat session with a customer."""

    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.lang = "en"  # default until user picks
        self.lang_chosen = False
        self.data = None

        all_data = _load_customer_data()
        if customer_id in all_data.index:
            self.data = all_data.loc[customer_id].to_dict()
        else:
            raise ValueError(f"Unknown customer_id: {customer_id}")

    # ---------------------------------------------------------------
    def greet(self) -> str:
        msg = t("greeting", self.lang)
        msg += "\n\n" + t("ask_language", self.lang)
        return msg

    # ---------------------------------------------------------------
    def handle(self, user_message: str) -> str:
        """Main entry point. Takes user text, returns bot reply."""

        # 1. Language choice — before anything else
        if not self.lang_chosen:
            pick = user_message.strip()
            if pick in LANGUAGES:
                self.lang = LANGUAGES[pick]["code"]
                self.lang_chosen = True
                return t("language_set", self.lang) + "\n\n" + t("ask_intent", self.lang)
            # If they didn't pick a number, ask again
            return t("ask_language", self.lang)

        # 2. Intent routing
        intent = detect_intent(user_message)
        return self._route(intent, user_message)

    # ---------------------------------------------------------------
    def _route(self, intent: str, raw: str) -> str:
        if intent == "greet":
            return t("greeting", self.lang) + "\n" + t("ask_intent", self.lang)

        if intent == "help":
            return t("help_response", self.lang)

        if intent == "balance":
            return t("balance_response", self.lang,
                     income=self.data.get("monthly_declared_income", 0),
                     savings=self.data.get("savings_rate_mean_7", 0) or 0)

        if intent == "emi":
            emi = self.data.get("emi_sum_mean_7", 0) or 0
            ratio = self.data.get("emi_to_income_mean_7", 0) or 0
            return t("emi_response", self.lang, emi=emi, ratio=ratio)

        if intent == "upi":
            share = self.data.get("upi_share_mean_7", 0) or 0
            return t("upi_response", self.lang, share=share)

        if intent == "loan":
            return self._handle_loan()

        if intent == "fraud":
            return self._handle_fraud()

        if intent == "agent":
            return "Connecting you to a human agent... (demo)"

        if intent == "bye":
            return t("farewell", self.lang)

        return t("fallback", self.lang)

    # ---------------------------------------------------------------
    def _handle_loan(self) -> str:
        """Guardrail-aware loan response."""

        # STRICT GUARDRAIL — never offer loans to high-stress customers
        if self.data.get("stress_level") == "high":
            return t("loan_suppressed", self.lang)

        # Recommend the product the customer has shown most interest in
        interests = {
            "personal": self.data.get("interest_personal", 0) or 0,
            "home":     self.data.get("interest_home", 0) or 0,
            "auto":     self.data.get("interest_auto", 0) or 0,
            "mortgage": self.data.get("interest_mortgage", 0) or 0,
        }
        top = max(interests, key=interests.get)
        if interests[top] == 0:
            # fall through to default
            top = "personal"

        # Naive amount: 12 months income for personal, 60 months for home
        income = self.data.get("monthly_declared_income", 0) or 0
        multiplier = {"personal": 12, "auto": 24, "home": 60, "mortgage": 36}[top]
        amount = income * multiplier

        return t("loan_offer_safe", self.lang, product=top, amount=amount)

    # ---------------------------------------------------------------
    def _handle_fraud(self) -> str:
        return ("I understand your concern. Let me connect you with our fraud "
                "response team immediately. Please do not share OTPs with anyone. "
                "(demo)")

    # ---------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "language": self.lang,
            "stress_level": self.data.get("stress_level"),
            "is_stressed": self.data.get("is_stressed"),
        }
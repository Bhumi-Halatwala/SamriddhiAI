"""
tests/test_chatbot.py
Phase 6 tests for the chatbot.
Run from SamriddhiAI/ root:
    py -3.11 -m pytest tests/test_chatbot.py -v
"""

import pandas as pd
import pytest
from pathlib import Path

from src.chatbot.router import ChatSession
from src.chatbot.intents import detect_intent

PROCESSED = Path("data/processed")


@pytest.fixture(scope="module")
def safe_customer():
    s = pd.read_parquet(PROCESSED / "stress_features.parquet")
    return s[s["stress_level"] == "low"].iloc[0]["customer_id"]


@pytest.fixture(scope="module")
def stressed_customer():
    s = pd.read_parquet(PROCESSED / "stress_features.parquet")
    return s[s["stress_level"] == "high"].iloc[0]["customer_id"]


def test_intent_greet():
    assert detect_intent("hello") == "greet"
    assert detect_intent("नमस्ते") == "greet"


def test_intent_loan():
    assert detect_intent("I want a loan") == "loan"
    assert detect_intent("कर्ज") == "loan"


def test_intent_balance():
    assert detect_intent("what is my balance") == "balance"


def test_intent_emi():
    assert detect_intent("EMI details please") == "emi"


def test_intent_help():
    assert detect_intent("help me") == "help"


def test_intent_unknown():
    assert detect_intent("random gibberish xyz") == "unknown"


def test_language_must_be_chosen_first(safe_customer):
    s = ChatSession(safe_customer)
    reply = s.handle("hello")
    assert "1)" in reply or "choose" in reply.lower() or "चुनें" in reply
    assert s.lang_chosen is False


def test_language_set_after_choice(safe_customer):
    s = ChatSession(safe_customer)
    s.handle("2")
    assert s.lang_chosen is True
    assert s.lang == "hi"


def test_stressed_customer_loan_suppressed(stressed_customer):
    s = ChatSession(stressed_customer)
    s.handle("1")
    reply = s.handle("I want a loan")
    assert "advisor" in reply.lower() or "commitments" in reply.lower()
    assert "loan could suit" not in reply.lower()


def test_safe_customer_can_get_loan_offer(safe_customer):
    s = ChatSession(safe_customer)
    s.handle("1")
    reply = s.handle("I want a loan")
    assert any(w in reply.lower() for w in ["personal", "home", "auto", "mortgage", "₹"])


def test_hindi_greeting(safe_customer):
    s = ChatSession(safe_customer)
    s.handle("2")
    reply = s.handle("मुझे मदद चाहिए")
    assert len(reply) > 0
    reply_en = ChatSession(safe_customer)
    reply_en.handle("1")
    reply_en_out = reply_en.handle("help")
    assert reply != reply_en_out


def test_farewell(safe_customer):
    s = ChatSession(safe_customer)
    s.handle("1")
    reply = s.handle("bye")
    assert "thank" in reply.lower() or "day" in reply.lower()

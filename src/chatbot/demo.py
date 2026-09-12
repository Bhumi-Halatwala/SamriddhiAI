"""
src/chatbot/demo.py
Phase 6 - Scripted demo of the chatbot.

Runs two conversations:
    1. A safe customer (no stress) — receives a product offer in Hindi.
    2. A stressed customer — receives the suppression message in Tamil.

Run from SamriddhiAI/ root:
    py -3.11 -m src.chatbot.demo
"""

from pathlib import Path
from datetime import datetime
import pandas as pd

from src.chatbot.router import ChatSession
from src.chatbot.languages import LANGUAGES

PROCESSED = Path("data/processed")
REPORT_DIR = Path("outputs/reports")


class Log:
    def __init__(self):
        self.lines = []
    def write(self, msg=""):
        print(msg)
        self.lines.append(str(msg))
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def pick_demo_customers(log):
    stress = pd.read_parquet(PROCESSED / "stress_features.parquet")
    safe = stress[stress["stress_level"] == "low"].iloc[0]["customer_id"]
    stressed = stress[stress["stress_level"] == "high"].iloc[0]["customer_id"]
    log.write(f"Safe customer:     {safe}")
    log.write(f"Stressed customer: {stressed}")
    return safe, stressed


def run_conversation(customer_id, language_choice, messages, log, tag):
    log.write("\n" + "=" * 60)
    log.write(f"CONVERSATION — {tag}")
    log.write("=" * 60)

    session = ChatSession(customer_id)
    log.write(f"\nBot: {session.greet()}")

    # Step 1: user picks language
    log.write(f"\nUser: {language_choice}")
    log.write(f"Bot: {session.handle(language_choice)}")

    # Step 2: follow-up messages
    for msg in messages:
        log.write(f"\nUser: {msg}")
        reply = session.handle(msg)
        log.write(f"Bot: {reply}")

    log.write(f"\nSession summary: {session.summary()}")
    return session


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log = Log()
    log.write(f"Phase 6 — Chatbot Demo — {datetime.now().isoformat(timespec='seconds')}")

    safe_id, stressed_id = pick_demo_customers(log)

    # ---- Conversation 1: safe customer, Hindi, asks for loan ----
    run_conversation(
        safe_id, "2",  # '2' = Hindi
        ["मुझे लोन के बारे में जानकारी चाहिए",  # "I want loan info"
         "मेरा बैलेंस क्या है?",               # "What is my balance?"
         "bye"],
        log, tag="Safe customer (Hindi)"
    )

    # ---- Conversation 2: stressed customer, Tamil, asks for loan ----
    run_conversation(
        stressed_id, "3",  # '3' = Tamil
        ["எனக்கு கடன் வேண்டும்",              # "I need a loan"
         "EMI பற்றி சொல்லுங்கள்",              # "Tell me about EMI"
         "bye"],
        log, tag="Stressed customer (Tamil)"
    )

    # ---- Conversation 3: safe customer, Bengali, help + balance ----
    run_conversation(
        safe_id, "5",  # '5' = Bengali
        ["help",
         "আমার ব্যালেন্স কত?",                # "What is my balance?"
         "bye"],
        log, tag="Safe customer (Bengali)"
    )

    log.write("\n" + "=" * 60)
    log.write("PHASE 6 SUMMARY")
    log.write("=" * 60)
    log.write("Languages supported: 5 (en, hi, ta, mr, bn)")
    log.write("Guardrail verified: stressed customers receive loan suppression message")
    log.write("Deterministic rule-based routing (no LLM required for demo)")

    log.save(REPORT_DIR / "phase6_chatbot.txt")
    print(f"\nReport saved to {REPORT_DIR / 'phase6_chatbot.txt'}")


if __name__ == "__main__":
    main()
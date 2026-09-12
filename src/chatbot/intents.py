"""
src/chatbot/intents.py
Phase 6 - Rule-based intent detection.

Simple regex/keyword rules — deterministic, explainable, and sufficient
for a demo without an LLM. In production this would be replaced by or
augmented with an LLM router.
"""

import re

# Intent patterns. Order matters: first match wins.
RULES = [
    ("greet",      [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"namaste",
                    r"नमस्ते", r"வணக்கம்", r"নমস্কার", r"नमस्कार"]),
    ("balance",    [r"balance", r"savings", r"बैलेंस", r"शिल्लक",
                    r"இருப்பு", r"ব্যালেন্স"]),
    ("loan",       [r"\bloan\b", r"\bloans\b", r"borrow", r"लोन", r"कर्ज",
                    r"கடன்", r"ঋণ"]),
    ("emi",        [r"\bemi\b", r"installment", r"ईएमआई", r"त installment",
                    r"ई.एम.आई"]),
    ("upi",        [r"\bupi\b", r"यूपीआई", r"यूपीआय"]),
    ("help",       [r"help", r"मदद", r"உதவி", r"সাহায্য"]),
    ("bye",        [r"bye", r"goodbye", r"अलविदा", r"विदाई", r"பை"]),
    ("agent",      [r"agent", r"human", r"representative", r"एजेंट",
                    r"முகவர்"]),
    ("fraud",      [r"fraud", r"unauthorized", r"suspicious", r"धोखा",
                    r"धोखाधड़ी", r"मोषक"]),
]


def detect_intent(message: str) -> str:
    """Return the intent label for a user message."""
    s = message.lower().strip()
    for intent, patterns in RULES:
        for p in patterns:
            if re.search(p, s, re.IGNORECASE):
                return intent
    return "unknown"
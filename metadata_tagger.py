"""
metadata_tagger.py
------------------
Rule-based intent labelling and outcome tagging for XYZ chatbot interactions and chunks.

Intent labels are used for:
  - Colour coding in UMAP / t-SNE visualizations
  - Metadata-filtered retrieval in the RAG chain
  - Measuring inter-document cluster separation (silhouette score)
"""

from __future__ import annotations
from typing import List
from data_parser import Interaction, Chunk


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

INTENT_RULES: list[tuple[str, list[str]]] = [
    # (intent_label, keyword_triggers)
    ("card_replacement",    ["new debit card", "replacement card", "lost", "stolen",
                             "card doesnt work", "card retained", "atm took"]),
    ("balance_enquiry",     ["balance", "what's my balance"]),
    ("password_reset",      ["forgot", "password", "re-register", "login"]),
    ("savings_account",     ["savings account", "open.*account"]),
    ("direct_debit",        ["direct debit", "sort code", "account number"]),
    ("fraud_alert",         ["suspicious", "text message", "fraud", "7726",
                             "phishing", "scam"]),
    ("branch_locator",      ["nearest branch", "branch locator", "atm",
                             "find.*branch", "send link"]),
    ("mortgage",            ["mortgage", "mortgage advisor"]),
    ("address_change",      ["change address", "change my address",
                             "how to change address", "sycamore"]),
    ("travel_notification", ["traveling", "abroad", "travel", "travel notifications"]),
]

OUTCOME_RULES: dict[str, list[str]] = {
    "resolved":   ["did i answer your question", "please come back any time",
                   "was that helpful"],
    "handoff":    ["connecting you with one of my colleagues",
                   "connecting you with", "handoff"],
    "loop":       ["i'm sorry i didn't understand", "please could you rephrase",
                   "ill have another go"],
}


# ---------------------------------------------------------------------------
# Tagging helpers
# ---------------------------------------------------------------------------

def _text_contains_any(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    import re
    for kw in keywords:
        if re.search(kw, text_lower):
            return True
    return False


def tag_intent(text: str) -> str:
    """Return the best matching intent label for the given text."""
    for intent_label, keywords in INTENT_RULES:
        if _text_contains_any(text, keywords):
            return intent_label
    return "general_enquiry"


def tag_outcome(text: str) -> str:
    """Return outcome tag: resolved | handoff | loop | unknown."""
    for outcome, keywords in OUTCOME_RULES.items():
        if _text_contains_any(text, keywords):
            return outcome
    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tag_interactions(interactions: List[Interaction]) -> List[Interaction]:
    """
    Mutates each interaction's metadata dict with:
      - intent:  primary intent label
      - outcome: resolved | handoff | loop | unknown
    Returns the same list for chaining.
    """
    for ia in interactions:
        full_text = ia.full_text
        ia.__dict__.setdefault("intent", tag_intent(full_text))
        ia.__dict__.setdefault("outcome", tag_outcome(full_text))
    return interactions


def tag_chunks(chunks: List[Chunk], interactions: List[Interaction]) -> List[Chunk]:
    """
    Enriches each chunk's metadata with intent and outcome.
    - Interaction-level chunks  → label from the full interaction text
    - Turn-window chunks        → label from the window text
    Also adds interaction-level outcome to turn chunks for richer filtering.
    """
    # Build a lookup: interaction_id → (intent, outcome)
    ia_lookup = {}
    for ia in interactions:
        intent = tag_intent(ia.full_text)
        outcome = tag_outcome(ia.full_text)
        ia_lookup[ia.id] = (intent, outcome)

    for chunk in chunks:
        if chunk.chunk_type == "interaction":
            intent, outcome = ia_lookup.get(chunk.interaction_id, ("general_enquiry", "unknown"))
            chunk.metadata["intent"] = intent
            chunk.metadata["outcome"] = outcome
        else:  # turn_window
            # Local intent from the window text itself (more specific)
            local_intent = tag_intent(chunk.text)
            _, outcome = ia_lookup.get(chunk.interaction_id, ("general_enquiry", "unknown"))
            chunk.metadata["intent"] = local_intent
            chunk.metadata["outcome"] = outcome

    return chunks


# Colour palette for visualization
INTENT_COLORS = {
    "card_replacement":    "#e74c3c",
    "balance_enquiry":     "#3498db",
    "password_reset":      "#9b59b6",
    "savings_account":     "#2ecc71",
    "direct_debit":        "#f39c12",
    "fraud_alert":         "#e67e22",
    "branch_locator":      "#1abc9c",
    "mortgage":            "#34495e",
    "address_change":      "#c0392b",
    "travel_notification": "#16a085",
    "general_enquiry":     "#95a5a6",
}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from data_parser import parse_interactions, build_chunks

    input_path = Path(__file__).parent / "Input Decs" / "input.txt"
    interactions = parse_interactions(input_path)
    chunks = build_chunks(interactions)

    chunks = tag_chunks(chunks, interactions)

    print("Intent & outcome tagging results:\n")
    for chunk in chunks:
        if chunk.chunk_type == "interaction":
            print(
                f"  Interaction {chunk.interaction_id:>2d} | "
                f"intent={chunk.metadata['intent']:<22s} | "
                f"outcome={chunk.metadata['outcome']}"
            )

"""
Input validation guard — runs before any ML code in the search handler.

Returns (True, "") if valid.
Returns (False, user-readable error message) if invalid.

Catches:
  1. Empty / whitespace only
  2. Too short (< QUERY_MIN_CHARS characters)
  3. Too few words (< QUERY_MIN_WORDS words)
  4. Too long (> QUERY_MAX_CHARS characters)
  5. Non-Latin script (Hindi, Tamil, etc.) — model is English-only
  6. No legal signal words — not a legal query
"""

import re

# ── Constants defined locally — do NOT import from config ─────────────────
# Importing from config can fail silently if .env is missing,
# causing the validator to pass everything. Define here explicitly.
QUERY_MIN_CHARS = 20
QUERY_MAX_CHARS = 5000
QUERY_MIN_WORDS = 4

# ── Legal signal words ─────────────────────────────────────────────────────
# At least ONE of these must appear in a valid legal query.
# Sorted roughly by frequency in Indian legal text.
LEGAL_SIGNALS = {
    "ipc", "section", "accused", "court", "bail", "murder", "rape",
    "fraud", "appeal", "conviction", "acquittal", "sentence", "judge",
    "petitioner", "respondent", "plaintiff", "defendant", "fir", "charge",
    "arrest", "custody", "evidence", "witness", "verdict", "judgment",
    "judgement", "crpc", "article", "writ", "habeas", "injunction",
    "decree", "theft", "robbery", "assault", "cheating", "dacoity",
    "offence", "offense", "convicted", "acquitted", "penalty", "parole",
    "prosecution", "defence", "defense", "sessions", "magistrate",
    "high court", "supreme court", "tribunal", "hearing", "order",
}

# ── Non-Latin script detection ─────────────────────────────────────────────
# Compiled once at module load. Matches any Indic script character.
_NON_LATIN_RE = re.compile(
    "["
    "\u0900-\u097F"  # Devanagari (Hindi, Marathi, Sanskrit)
    "\u0980-\u09FF"  # Bengali
    "\u0A00-\u0A7F"  # Gurmukhi (Punjabi)
    "\u0A80-\u0AFF"  # Gujarati
    "\u0B00-\u0B7F"  # Odia
    "\u0B80-\u0BFF"  # Tamil
    "\u0C00-\u0C7F"  # Telugu
    "\u0C80-\u0CFF"  # Kannada
    "\u0D00-\u0D7F"  # Malayalam
    "\u0E00-\u0E7F"  # Thai
    "\u0600-\u06FF"  # Arabic
    "]"
)


def validate_query(text: str) -> tuple:
    """
    Validate a search query before sending to the ML pipeline.

    Args:
        text: raw query string from user input

    Returns:
        (True, "")                              if valid
        (False, human-readable error message)   if invalid
    """
    # ── 1. Empty check ─────────────────────────────────────────────────────
    if not text or not text.strip():
        return False, (
            "Please describe your case. The search field is empty."
        )

    text_stripped = text.strip()

    # ── 2. Too long ────────────────────────────────────────────────────────
    # Check BEFORE splitting words — a 10,000-char string is expensive to split
    if len(text_stripped) > QUERY_MAX_CHARS:
        return False, (
            f"Query too long ({len(text_stripped):,} characters — "
            f"limit is {QUERY_MAX_CHARS:,}). "
            f"Please summarize the key charges, facts, and evidence "
            f"in a few sentences. For a full judgment text, use the PDF upload feature."
        )

    # ── 3. Too short (character count) ─────────────────────────────────────
    if len(text_stripped) < QUERY_MIN_CHARS:
        return False, (
            f"Query too short ({len(text_stripped)} characters — "
            f"minimum is {QUERY_MIN_CHARS}). "
            f"Example: 'Accused charged under IPC Section 302 for murder "
            f"with eyewitness and forensic evidence.'"
        )

    # ── 4. Too brief (word count) ──────────────────────────────────────────
    words = text_stripped.split()
    if len(words) < QUERY_MIN_WORDS:
        return False, (
            f"Query too brief ({len(words)} words). "
            f"Please describe the charges, facts, and evidence "
            f"in at least {QUERY_MIN_WORDS} words."
        )

    # ── 5. Non-Latin script detection ─────────────────────────────────────
    non_latin_chars = _NON_LATIN_RE.findall(text_stripped)
    # If more than 20% of characters are non-Latin, reject
    if len(non_latin_chars) > len(text_stripped) * 0.20:
        return False, (
            f"Query appears to be in a non-English script "
            f"({len(non_latin_chars)} non-Latin characters detected). "
            f"LexAI's embedding model (LegalBERT) was trained on English legal text. "
            f"Please enter your query in English for accurate results."
        )

    # ── 6. Legal signal check ──────────────────────────────────────────────
    text_lower = text_stripped.lower()
    has_signal = any(signal in text_lower for signal in LEGAL_SIGNALS)

    if not has_signal:
        return False, (
            "Query doesn't appear to describe a legal case. "
            "Please include legal context such as charges (IPC section number), "
            "case type (murder, bail, fraud, appeal), court name, or parties involved. "
            "Example: 'Accused charged under IPC 420 for cheating and fraud. "
            "FIR filed by complainant in Sessions Court.'"
        )

    return True, ""

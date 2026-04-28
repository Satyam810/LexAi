"""Query validator — sanitize and validate user search input."""
from config import QUERY_MIN_CHARS, QUERY_MAX_CHARS, QUERY_MIN_WORDS
import re


def validate_query(query: str) -> tuple:
    """
    Validate a search query.

    Returns:
        (is_valid: bool, clean_query: str, error_message: str)
    """
    if not query or not isinstance(query, str):
        return False, "", "Query cannot be empty."

    # Strip and normalize whitespace
    clean = re.sub(r'\s+', ' ', query.strip())

    # Remove dangerous characters (basic XSS prevention)
    clean = re.sub(r'[<>{}]', '', clean)

    if len(clean) < QUERY_MIN_CHARS:
        return False, clean, (
            f"Query too short ({len(clean)} chars). "
            f"Minimum {QUERY_MIN_CHARS} characters required. "
            f"Try adding more detail about the case."
        )

    if len(clean) > QUERY_MAX_CHARS:
        clean = clean[:QUERY_MAX_CHARS]

    word_count = len(clean.split())
    if word_count < QUERY_MIN_WORDS:
        return False, clean, (
            f"Query too short ({word_count} words). "
            f"Minimum {QUERY_MIN_WORDS} words required. "
            f"Example: 'bail application under IPC 302 murder'"
        )

    return True, clean, ""

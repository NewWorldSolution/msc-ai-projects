"""Tolerant exact-text comparison shared by the Response and Knowledge validators."""

# `str.maketrans` builds a translation table from code point to replacement.
_QUOTE_GLYPHS = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
})


# Credentials are never a lookup target or a customer question; a how-to answer never depends on them.
SECRET_NOUNS = ("password", "passcode", "pin", "otp", "one-time code", "security code", "verification code",
                "security answer", "credential", "credentials", "login details", "account verification details")
# Information *about* a credential process (an error on the reset screen) is not a secret request.
PROCESS_WORDS = ("error", "message", "screen", "page", "step", "steps", "process", "link", "button",
                 "reset email", "requirement", "policy", "venue")


def mentions_credential(text: str) -> bool:
    """True when the text asks for a secret value, not for information about a credential process."""
    import re
    lowered = normalize_text(text)
    # \b marks a word boundary, so "pin" does not match "pinned".
    if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in PROCESS_WORDS):
        return False
    return any(re.search(rf"\b{re.escape(noun)}\b", lowered) for noun in SECRET_NOUNS)


def normalize_text(text: str) -> str:
    """Return text with straight quotes, single spaces and casefolded letters.

    Only comparisons use this value. Stored drafts, citations and the fixed
    error messages keep the original wording.
    """
    if not isinstance(text, str):
        raise TypeError("normalize_text requires a string.")
    # `split()` with no argument breaks on any whitespace run; join restores one space.
    return " ".join(text.translate(_QUOTE_GLYPHS).split()).casefold()

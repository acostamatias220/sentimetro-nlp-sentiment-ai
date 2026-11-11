"""Text normalization shared by training and serving.

Keeping one function for both sides avoids *training/serving skew*: the model
always sees text cleaned exactly the same way it was cleaned during training.
It is intentionally light: transformers handle casing and punctuation well,
and the TF-IDF baseline lowercases on its own.
"""

from __future__ import annotations

import html
import re
import unicodedata

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML_TAG_RE = re.compile(r"<[^>]{1,100}>")
_REPEAT_RE = re.compile(r"(.)\1{3,}")
_SPACE_RE = re.compile(r"\s+")
_CONTROL_CATEGORIES = {"Cc", "Cf"}


def clean_text(text: str) -> str:
    """Normalize a raw review.

    - Unicode NFC (``"é"`` written as ``e`` + combining accent becomes one char).
    - Unescape HTML entities and drop tags (``<br />`` is common in reviews).
    - Replace URLs and e-mails by placeholders (no signal, and avoids leaking PII).
    - Collapse 4+ repeated characters to 3 (``"buenooooo"`` -> ``"buenooo"``),
      keeping the emphasis but shrinking the vocabulary.
    - Remove control characters and collapse whitespace.
    """
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" <url> ", text)
    text = _EMAIL_RE.sub(" <email> ", text)
    text = "".join(
        " " if unicodedata.category(ch) in _CONTROL_CATEGORIES else ch for ch in text
    )
    text = _REPEAT_RE.sub(r"\1\1\1", text)
    return _SPACE_RE.sub(" ", text).strip()

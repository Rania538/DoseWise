"""
Utility for text and drug name normalization.
"""
import pandas as pd
import unicodedata
import re


def normalize_text(s: str) -> str:
    """Normalize text for consistent drug matching.
    Strips diacritics/punctuation while preserving Arabic characters.
    """
    if pd.isna(s):
        return ''
    s = str(s)
    s = unicodedata.normalize('NFKC', s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    # remove punctuation but keep Arabic letters
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

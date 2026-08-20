import pandas as pd
import unicodedata
import re
from difflib import get_close_matches
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None
from pathlib import Path
import json


def normalize_text(s: str) -> str:
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


class DrugNormalizer:
    def __init__(self, uniq_csv_path: str = None):
        base = Path('d:/Documents/tessst')
        path = Path(uniq_csv_path) if uniq_csv_path else base / 'data' / 'processed' / 'unique_drug_names.csv'
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            self.normalized_names = [normalize_text(x) for x in df['normalized'].fillna('')]
            self.variants = {normalize_text(row['normalized']): json.loads(row['variants']) for _, row in df.iterrows()}
        else:
            self.normalized_names = []
            self.variants = {}

    def _score(self, a: str, b: str) -> float:
        # Use RapidFuzz if available for more robust fuzzy scores
        if fuzz:
            scores = [fuzz.ratio(a, b), fuzz.partial_ratio(a, b), fuzz.token_sort_ratio(a, b)]
            return max(scores) / 100.0
        else:
            # fallback to SequenceMatcher
            from difflib import SequenceMatcher
            return SequenceMatcher(None, a, b).ratio()

    def is_arabic(self, s: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", s))

    def arabizi_to_arabic(self, s: str) -> str:
        # common Arabizi digit→Arabic mappings
        mapping = {'2': 'أ', '3': 'ع', '4': 'ش', '5': 'خ', '6': 'ط', '7': 'ح', '8': 'غ', '9': 'ق'}
        return ''.join(mapping.get(ch, ch) for ch in s)

    def basic_arabic_to_latin(self, s: str) -> str:
        # very small transliteration table to improve cross-script matching
        table = {
            'ا': 'a', 'أ': 'a', 'إ': 'i', 'آ': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h',
            'خ': 'kh', 'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'd',
            'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm',
            'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y', 'ى': 'a', 'ء': 'a', 'ؤ': 'u', 'ئ': 'y'
        }
        out = []
        for ch in s:
            out.append(table.get(ch, ch))
        return normalize_text(''.join(out))

    def candidates(self, user_input: str, top_n: int = 5):
        q = normalize_text(user_input)
        if not q:
            return []
        # if input looks like Arabizi (contains digits used as Arabic letters), try converting
        alt_q = None
        if re.search(r"[0-9]", user_input):
            conv = self.arabizi_to_arabic(user_input)
            alt_q = self.basic_arabic_to_latin(conv)
        elif self.is_arabic(user_input):
            alt_q = self.basic_arabic_to_latin(user_input)

        scores = []
        for name in self.normalized_names:
            sc = self._score(q, name)
            if alt_q:
                sc = max(sc, self._score(alt_q, name))
            scores.append((name, sc))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def identify(self, user_input: str, threshold: float = 0.75):
        q = normalize_text(user_input)
        if not q:
            return {'match': None, 'confidence': 0.0, 'candidates': []}
        candidates = self.candidates(q, top_n=10)
        best = candidates[0] if candidates else (None, 0.0)
        match, score = best
        if score >= threshold:
            return {'match': match, 'confidence': float(score), 'variants': self.variants.get(match, [])}
        else:
            # return top suggestions
            return {'match': None, 'confidence': float(score), 'candidates': candidates}


if __name__ == '__main__':
    dn = DrugNormalizer()
    tests = ['Abacavir', 'abacavir', 'Abacvir', 'Dolutegravir', 'dolutegravir', 'Panadol', 'بنادول']
    for t in tests:
        res = dn.identify(t)
        print(t, '->', res)

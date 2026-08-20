import requests
import json
import re
from pathlib import Path
from typing import Optional

CACHE_PATH = Path('d:/Documents/tessst/data/cache/api_cache.json')
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
if not CACHE_PATH.exists():
    CACHE_PATH.write_text(json.dumps({}))


def _load_cache():
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _write_cache(c):
    CACHE_PATH.write_text(json.dumps(c, ensure_ascii=False, indent=2))


def query_rxnav(term: str) -> Optional[dict]:
    """Query RxNav (RxNorm) for a term. Returns parsed JSON or None on failure.
    Caches results locally to avoid repeated requests.
    """
    cache = _load_cache()
    key = f"rxnav::{term}"
    if key in cache:
        return cache[key]

    url = f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={requests.utils.quote(term)}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            cache[key] = data
            _write_cache(cache)
            return data
    except Exception:
        return None
    return None


def _clean_name_for_lookup(name: str) -> str:
    if not name:
        return ''
    name = str(name).strip()
    name = re.sub(r'\s*\[.*?\]\s*$', '', name)
    return name.strip()


def _extract_ingredient_parts(name: str):
    """Convert a product name like 'abacavir 600 MG / dolutegravir 50 MG ...'
    into ingredient pieces like ['abacavir', 'dolutegravir'].
    """
    if not name:
        return []

    cleaned = _clean_name_for_lookup(name)
    if not cleaned:
        return []

    candidates = re.split(r'\s*/\s*|\s*\+\s*', cleaned)
    ingredients = []

    for part in candidates:
        p = part.strip()
        if not p:
            continue

        p = re.sub(r'^[\[\(\{]?\s*\d+(?:\.\d+)?\s*(?:HR|MG|G|ML|MCG|UG|MEQ|IU|UNITS?)?\s*[\]\)\}]?\s*', '', p, flags=re.I)
        p = re.sub(r'\s+\d+(?:\.\d+)?\s*(?:MG|G|ML|MCG|UG|MEQ|%|IU|UNITS?)\b.*$', '', p, flags=re.I)
        p = re.sub(r'\s+(?:ORAL|TABLET|CAPSULE|SOLUTION|SUSPENSION|INJECTION|TOPICAL|CHEWABLE|EXTENDED|DELAYED|RELEASE|DROPS|PACK|FORMULA)\b.*$', '', p, flags=re.I)
        p = re.sub(r'^[\[\(\{]+|[\]\)\}]+$', '', p)
        p = re.sub(r'\s+', ' ', p).strip(' ,;')

        if p and re.search(r'[A-Za-z\u0600-\u06FF]', p) and not re.fullmatch(r'[\d\s\-/\[\]\(\)\{\}]+', p):
            ingredients.append(p)

    # As a fallback, if no ingredient was extracted, keep the product name minus dosage suffixes.
    if not ingredients:
        fallback = re.sub(r'\s+\d+(?:\.\d+)?\s*(?:MG|G|ML|MCG|UG|MEQ|%|IU|UNITS?)\b.*$', '', cleaned, flags=re.I)
        fallback = re.sub(r'\s+(?:ORAL|TABLET|CAPSULE|SOLUTION|SUSPENSION|INJECTION|TOPICAL|CHEWABLE|EXTENDED|DELAYED|RELEASE|DROPS|PACK)\b.*$', '', fallback, flags=re.I)
        fallback = re.sub(r'\s+', ' ', fallback).strip(' ,;')
        if fallback:
            ingredients = [fallback]

    return ingredients


def lookup_drug(term: str) -> dict:
    """High-level lookup that tries RxNav and returns normalized fields.
    Returns a dict with possible keys: brand_names, generic_name, ingredients, sources
    """
    result = {'term': term, 'brand_names': [], 'generic_name': None, 'ingredients': [], 'sources': []}
    rx = query_rxnav(term)
    if rx:
        result['sources'].append('rxnav')
        try:
            concepts = rx.get('drugGroup', {}).get('conceptGroup', [])
            seen_brands = set()
            seen_ingredients = set()
            for cg in concepts:
                ts = cg.get('conceptProperties') or []
                if not ts:
                    continue
                for c in ts:
                    name = c.get('name')
                    if name:
                        clean_name = _clean_name_for_lookup(name)
                        if clean_name and clean_name.lower() not in seen_brands:
                            result['brand_names'].append(clean_name)
                            seen_brands.add(clean_name.lower())

                        for ingredient in _extract_ingredient_parts(name):
                            ingredient_key = ingredient.lower()
                            if ingredient_key and ingredient_key not in seen_ingredients:
                                seen_ingredients.add(ingredient_key)
                                result['ingredients'].append(ingredient)

            if result['ingredients']:
                normalized_term = term.strip().lower()
                for ingredient in result['ingredients']:
                    low_ingredient = ingredient.lower()
                    if low_ingredient == normalized_term or normalized_term in low_ingredient:
                        result['generic_name'] = ingredient
                        break
                if not result['generic_name']:
                    result['generic_name'] = result['ingredients'][0]

            if result['generic_name'] is None and term:
                result['generic_name'] = term.strip()

        except Exception:
            pass

    return result


if __name__ == '__main__':
    print('Cache path:', CACHE_PATH)
    print('Sample lookup (not executing call here).')

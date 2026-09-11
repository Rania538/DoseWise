import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

from src.drug_resolver import normalize_text, resolve_drug, SAFE_ALIASES

ROOT = Path(__file__).resolve().parents[1]
MASTER_INTERACTIONS_PATH = ROOT / 'data' / 'processed' / 'master_interactions.csv'
UNIQUE_NAMES_PATH = ROOT / 'data' / 'processed' / 'unique_drug_names.csv'


def detect_language_style(text: str) -> str:
    """Classify prose style while ignoring verified medication names."""
    if not text:
        return 'english'

    # Medication names are evidence for neither English nor Franco. Remove
    # known names before counting script and language markers.
    language_text = normalize_text(text)
    medication_names = set(normalize_text(name) for name in SAFE_ALIASES)
    try:
        medication_names.update(normalize_text(name) for name in _load_unique_names())
    except Exception:
        pass
    for name in sorted((name for name in medication_names if name), key=len, reverse=True):
        language_text = re.sub(r'(?<!\w)' + re.escape(name) + r'(?!\w)', ' ', language_text)
    language_text = re.sub(r'\s+', ' ', language_text).strip()

    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', language_text))
    latin_words = re.findall(r'[A-Za-z]+', language_text)
    
    # Common Arabizi words without distinguishing numerals
    arabizi_dict = {
        'mfhmsh', 'msh', 'tmam', 'eh', 'ezay', 'leh', 'keda', 'momken', 
        'kwayes', 'ana', 'enta', 'enti', 'izayak', 'ezayak', 'katbly', 
        'doktor', 'tany', 'hom', 'ma3', 'wallahi', 'yb2a', 'khod', 'dawa', 'el'
    }
    
    arabizi_markers = re.findall(
        r'\b(?:' + '|'.join(arabizi_dict) + r')\b', 
        language_text.casefold()
    )
    
    has_arabizi_digits = bool(re.search(
        r'\b(?=[A-Za-z0-9]*[A-Za-z])\w*[2356789]\w*\b',
        language_text,
    ))
    
    if arabic_chars:
        return 'mixed' if latin_words or arabizi_markers or has_arabizi_digits else 'arabic'
        
    if has_arabizi_digits or arabizi_markers:
        return 'arabizi'
        
    if len(language_text.split()) <= 1 and not arabizi_markers and not has_arabizi_digits:
        if language_text.casefold() not in {
            'hi', 'hello', 'hey', 'thanks', 'okay', 'ok', 'yes', 'yeah', 'yep',
            'no', 'nah', 'nope', 'y', 'n',
        }:
            return 'arabic'
            
    return 'english'


def detect_language(text: str) -> str:
    """Return the legacy API language code while exposing the richer style separately."""
    return 'ar' if detect_language_style(text) in ('arabic', 'arabizi', 'mixed') else 'en'


_UNIQUE_NAMES_CACHE = None

def _load_unique_names() -> List[str]:
    global _UNIQUE_NAMES_CACHE
    if _UNIQUE_NAMES_CACHE is not None:
        return _UNIQUE_NAMES_CACHE
    if not UNIQUE_NAMES_PATH.exists():
        return []
    df = pd.read_csv(UNIQUE_NAMES_PATH, dtype=str)
    _UNIQUE_NAMES_CACHE = df['normalized'].dropna().astype(str).tolist()
    return _UNIQUE_NAMES_CACHE


def parse_medication_followup(text: str) -> Tuple[bool, List[str]]:
    """Check if the text is explicitly a follow-up asking about a medication.

    Returns (True, [medication_names]) when the message is a follow-up pattern
    like "what about simvastatin?", "طب و simvastatin؟", "و simvastatin؟".
    Returns (False, []) for pure conversational text.
    
    Prefixes are ordered longest-first so "وماذا عن" matches before "و".
    """
    # Longest-first to prevent short prefixes from consuming longer ones.
    prefixes = [
        # Arabic multi-word
        "وماذا عن", "ماذا عن", "طب و", "طيب و", "طب", "طيب",
        # English multi-word
        "what about", "how about", "and what about",
        # Short connectors (must come last)
        "and",
        # Arabic single-char connector و — handled specially below
    ]
    t_lower = text.strip().lower()
    t_clean = re.sub(r'[؟\?\.\!،,]', '', t_lower).strip()

    for prefix in prefixes:
        # Check "prefix + space + candidate" or exact match
        if t_clean.startswith(prefix + " "):
            med_candidate = t_clean[len(prefix):].strip()
            if not med_candidate:
                continue
            if len(med_candidate.split()) <= 4:
                extracted = extract_medications(med_candidate)
                if extracted:
                    # If it's purely Arabic, we must be careful not to catch conversational text.
                    # We accept it if it has Latin characters OR if it's known in aliases/unique names.
                    valid_extracted = []
                    known = set(normalize_text(k) for k in SAFE_ALIASES.keys())
                    for n in _load_unique_names():
                        known.add(normalize_text(n))
                    for ex in extracted:
                        ex_norm = normalize_text(ex)
                        if re.search(r'[A-Za-z]', ex) or ex_norm in known:
                            valid_extracted.append(ex)
                        elif ex_norm.startswith('و') and ex_norm[1:] in known:
                            valid_extracted.append(ex[1:])
                        elif ex_norm.startswith('وال') and ex_norm[3:] in known:
                            valid_extracted.append(ex[3:])
                        elif ex_norm.startswith('ال') and ex_norm[2:] in known:
                            valid_extracted.append(ex[2:])
                    if valid_extracted:
                        return True, valid_extracted
        elif t_clean == prefix:
            # Just the prefix alone with nothing after — not a follow-up
            continue

    # Special handling for Arabic و (waw al-atf) which may appear:
    #   - "و simvastatin" (with space)
    #   - "وsimvastatin"  (without space, common in Arabic typing)
    # But must NOT match "وأنت" or other pure Arabic conversational words.
    if t_clean.startswith("و"):
        remainder = t_clean[1:].strip()
        if remainder and len(remainder.split()) <= 4:
            extracted = extract_medications(remainder)
            if extracted:
                valid_extracted = []
                known = set(normalize_text(k) for k in SAFE_ALIASES.keys())
                try:
                    for n in _load_unique_names():
                        known.add(normalize_text(n))
                except Exception:
                    pass
                for ex in extracted:
                    ex_norm = normalize_text(ex)
                    if re.search(r'[A-Za-z]', ex) or ex_norm in known:
                        valid_extracted.append(ex)
                    elif ex_norm.startswith('ال') and ex_norm[2:] in known:
                        valid_extracted.append(ex[2:])
                if valid_extracted:
                    return True, valid_extracted

    return False, []


def extract_medications(text: str) -> List[str]:
    """Extract medication-like spans from free text using known aliases and a unique names list.

    Returns a list of unique user-provided spans (not normalized generics).
    """
    norm = normalize_text(text)
    candidates = []
    alias_variants = {}

    def alias_pattern(value: str) -> str:
        # Arabic text is commonly adjacent to punctuation or other Arabic words;
        # Unicode-aware lookarounds avoid the ASCII \b boundary edge cases.
        return r'(?<![\w\u0600-\u06ff])' + re.escape(normalize_text(value)) + r'(?![\w\u0600-\u06ff])'

    # check SAFE_ALIASES keys first (these are user-facing aliases/brands)
    for alias in sorted(SAFE_ALIASES.keys(), key=len, reverse=True):
        pattern = alias_pattern(alias)
        if re.search(pattern, norm):
            candidates.append(alias)
    for variant, alias in alias_variants.items():
        if re.search(alias_pattern(variant), norm):
            candidates.append(alias)

    # also scan the unique normalized names list for matches
    for name in _load_unique_names():
        name_norm = normalize_text(name)
        if not name_norm:
            continue
        pattern = r'\b' + re.escape(name_norm) + r'\b'
        if re.search(pattern, norm):
            candidates.append(name)

    # Deduplicate while preserving order of appearance in the original normalized text
    seen = set()
    ordered = []
    for c in candidates:
        key = normalize_text(c)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(c)
    # Heuristic token scanning: find short candidate phrases (1-3 words)
    # Avoid adding long sentence fragments. Only accept short phrases that are not
    # common stopwords or profession/sentence tokens.
    blacklist = set([
        'doctor', 'dr', 'neurologist', 'ent', 'gp', 'specialist', 'prescribed', 'prescribed by', 'wrote', 'took',
        'headache', 'pain', 'i', 'my', 'they', 'can', 'take', 'together', 'by', 'for', 'with', 'and', 'or',
        'visited', 'later', 'have', 'something', 'called', 'hello', 'hi', 'name', 'what', 'help', 'you', 'is'
    ])
    ar_blacklist = set([
        'دكتور', 'طبيب', 'صيدلي', 'كتب', 'كتبلي', 'دواء', 'عندي', 'خدت', 'ينفع', 'مع',
        'أنا', 'كنت', 'عند', 'بعدها', 'رحت', 'الأنف', 'الأذن', 'والأذن', 'وكتبلي'
        , 'اخد', 'أخد', 'اتنين', 'الاتنين', 'أخذ', 'خد', 'أخذت', 'اهلا', 'مرحبا', 'اسمي', 'ماذا', 'تساعدني', 'انت',
        'انا', 'باخد', 'هل', 'يمكنني', 'يمكن', 'تناول', 'تناولها', 'أقدر', 'اقدر', 'آخد', 'اخدهم', 'معهم',
        'من', 'هذا', 'هذه', 'الدواء', 'الدواء؟'
    ])
    arabizi_blacklist = set(['ana', 'kont', '3and', 'katbly', 'katab', 'doktor', 'dokter', 'tany', 'ynf3', 'akhodhom', 'm3', 'ba3d'])

    # capture 1-3 word sequences of letters/digits (including Arabic)
    for match in re.finditer(r"\b([A-Za-z\u0600-\u06FF0-9]{2,}(?:\s+[A-Za-z\u0600-\u06FF0-9]{2,}){0,2})\b", norm):
        span = match.group(1).strip()
        key = normalize_text(span)
        if not key or key in seen:
            continue
        if len(span) > 40:  # avoid long fragments
            continue
        # skip pure numeric tokens
        if re.fullmatch(r'\d+', span):
            continue
        parts = key.split()
        # if this short span is just a sequence of previously-seen tokens (e.g. "panadol amoxicillin"), skip it
        if len(parts) > 1 and all(p in seen for p in parts):
            continue
        if any(p in blacklist for p in parts):
            continue
        if any(p in ar_blacklist for p in parts):
            continue
        if any(p in arabizi_blacklist for p in parts):
            continue
        # require at least one alphabetic character
        if not re.search(r'[A-Za-z\u0600-\u06FF]', span):
            continue
        seen.add(key)
        ordered.append(span)

    # Preserve short unknown medication-like terms beside a medication connector
    # so the resolver can report them instead of silently dropping them.
    connector_re = re.compile(r'\b(?:with|and|مع|و)\b|\+')
    connector_stopwords = blacklist | ar_blacklist | arabizi_blacklist
    for connector in connector_re.finditer(norm):
        left = norm[:connector.start()].strip()
        right = norm[connector.end():].strip()
        for fragment, take_last in ((left, True), (right, False)):
            words = re.findall(r'[A-Za-z\u0600-\u06FF0-9]{2,}', fragment)
            if not words:
                continue
            candidate = words[-1] if take_last else words[0]
            candidate_key = normalize_text(candidate)
            if candidate_key in connector_stopwords or candidate_key in seen:
                continue
            if len(candidate_key) < 4 or len(candidate_key) > 32:
                continue
            seen.add(candidate_key)
            ordered.append(candidate)

    # Conservative 'X and Y' fallback: when only one side of a simple conjunction
    # was captured (e.g. 'XyzUnknown and amoxicillin'), try to add the missing
    # short token on the other side if it looks like a drug token and is not
    # blacklisted. This avoids forcing long sentences to resolver while still
    # capturing simple unknown tokens.
    if len(ordered) < 2 and ' and ' in norm:
        parts = re.split(r'\band\b', norm)
        if len(parts) >= 2:
            left, right = parts[0].strip(), parts[-1].strip()
            # pick the last word-like token from left
            m = re.search(r'([A-Za-z\u0600-\u06FF0-9-]{2,})\s*$', left)
            if m:
                cand = m.group(1).strip()
                cand_key = normalize_text(cand)
                if cand_key and cand_key not in seen:
                    if not re.fullmatch(r'\d+', cand) and re.search(r'[A-Za-z\u0600-\u06FF]', cand):
                        parts_c = cand_key.split()
                        if not any(p in blacklist for p in parts_c) and not any(p in ar_blacklist for p in parts_c) and not any(p in arabizi_blacklist for p in parts_c):
                            seen.add(cand_key)
                            ordered.insert(0, cand)

    # Final filter: drop spans that look like full sentences or are unusually long
    filtered = []
    full_norm = normalize_text(text)
    for s in ordered:
        s_norm = normalize_text(s)
        if not s_norm:
            continue
        # drop if it equals the full user message AND is longer than 2 words (avoid sending whole sentences)
        if s_norm == full_norm and len(s_norm.split()) > 2:
            continue
        # drop if more than 4 words or overly long
        if len(s_norm.split()) > 4:
            continue
        if len(s) > 120:
            continue
        filtered.append(s)

    return filtered

    # NOTE: fallback not reached; kept for clarity


def resolve_medications(med_spans: List[str]) -> List[Dict[str, Any]]:
    results = []
    # defensive pre-filter: ensure we only call resolver on short medication-like tokens
    for span in med_spans:
        if not span:
            continue
        s_norm = normalize_text(span)
        # skip if looks like a full sentence or contains question words
        if len(s_norm.split()) > 4:
            # return an ambiguous entry rather than call external resolver
            results.append({
                'input': span,
                'normalized_input': s_norm,
                'resolved_name': None,
                'generic_name': span,
                'active_ingredients': [],
                'rxcui': None,
                'match_type': 'no_match',
                'confidence': 0.0,
                'verified': False,
                'source': 'none',
                'reason': 'Span looks like a sentence; not sent to resolver.',
                'ddinter_mapping': {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []},
            })
            continue
        # skip obvious yes/no or short conversational tokens
        if s_norm in ('yes','yeah','yep','no','nah','لا','نعم','ايوه','أيوه','ايوه'):
            results.append({
                'input': span,
                'normalized_input': s_norm,
                'resolved_name': None,
                'generic_name': span,
                'active_ingredients': [],
                'rxcui': None,
                'match_type': 'no_match',
                'confidence': 0.0,
                'verified': False,
                'source': 'none',
                'reason': 'Conversational token; not sent to resolver.',
                'ddinter_mapping': {'ddinter_name': None, 'ddinter_rows': 0, 'matches': []},
            })
            continue

        res = resolve_drug(span, ddinter_csv_path=MASTER_INTERACTIONS_PATH)
        results.append(res)
    return results


def _norm_pair(a: str, b: str) -> Tuple[str, str]:
    return normalize_text(a), normalize_text(b)


_MASTER_INTERACTIONS_CACHE = None

def _get_master_interactions_df():
    global _MASTER_INTERACTIONS_CACHE
    if _MASTER_INTERACTIONS_CACHE is not None:
        return _MASTER_INTERACTIONS_CACHE
    _MASTER_INTERACTIONS_CACHE = pd.read_csv(MASTER_INTERACTIONS_PATH, dtype=str)
    return _MASTER_INTERACTIONS_CACHE

def retrieve_interactions(
    generic_names: List[str],
    resolved_medications: List[Dict[str, Any]] = None,
    new_medications: List[str] = None
) -> List[Dict[str, Any]]:
    """Given a list of verified generic names, produce evidence for each unique pair.

    Returns a list of evidence dicts per pair. Each matching DDInter row is an
    interaction-level chunk identified by its stable source row number.
    """
    pairs = []
    # unique pairs
    n = len(generic_names)
    if n < 2:
        return []

    df = _get_master_interactions_df()
    ingredient_lookup = {}
    for medication in resolved_medications or []:
        generic = medication.get('generic_name')
        if generic:
            ingredient_lookup[normalize_text(generic)] = medication.get('active_ingredients') or [generic]

    seen_pairs = set()
    
    if new_medications:
        new_medications_norm = {normalize_text(m) for m in new_medications}
    else:
        new_medications_norm = None

    for i in range(n):
        for j in range(i + 1, n):
            a = generic_names[i]
            b = generic_names[j]
            a_norm, b_norm = _norm_pair(a, b)
            
            if a_norm == b_norm:
                continue
                
            if new_medications_norm is not None:
                if a_norm not in new_medications_norm and b_norm not in new_medications_norm:
                    continue
                
            pair_key = frozenset([a_norm, b_norm])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            mask_ab = (df['drug_a'].map(lambda x: normalize_text(x)) == a_norm) & (df['drug_b'].map(lambda x: normalize_text(x)) == b_norm)
            mask_ba = (df['drug_a'].map(lambda x: normalize_text(x)) == b_norm) & (df['drug_b'].map(lambda x: normalize_text(x)) == a_norm)
            matches = df.loc[mask_ab | mask_ba]

            evidence = {
                'drug_a': a,
                'drug_b': b,
                'interaction_level': None,
                'source': 'DDInter',
                'ddinter_rows': 0,
                'matches': [],
                'chunk_ids': [],
                'source_rows': [],
                'active_ingredients': {
                    'drug_a': ingredient_lookup.get(a_norm, [a]),
                    'drug_b': ingredient_lookup.get(b_norm, [b]),
                },
            }

            if not matches.empty:
                # collect structured records
                rows = []
                for row_index, row in matches.iterrows():
                    source_row = int(row_index) + 2  # account for the CSV header
                    chunk_id = f'ddinter-row-{source_row}'
                    rows.append({
                        'chunk_id': chunk_id,
                        'source_row': source_row,
                        'drug_a': row['drug_a'],
                        'drug_b': row['drug_b'],
                        'interaction_level': row.get('interaction_level'),
                        'support_count': row.get('support_count'),
                        'ddinter_pairs': row.get('ddinter_pairs'),
                        'sample_origins_a': row.get('sample_origins_a'),
                        'sample_origins_b': row.get('sample_origins_b'),
                        'active_ingredients_a': ingredient_lookup.get(normalize_text(row['drug_a']), [row['drug_a']]),
                        'active_ingredients_b': ingredient_lookup.get(normalize_text(row['drug_b']), [row['drug_b']]),
                        'evidence': f'DDInter master_interactions.csv row {source_row}',
                    })
                # choose highest severity if multiple rows: Major>Moderate>Minor (if present)
                levels = [r.get('interaction_level') for r in rows if r.get('interaction_level')]
                priority = {'major': 3, 'moderate': 2, 'minor': 1}
                chosen = None
                if levels:
                    levels_norm = [l.lower() for l in levels]
                    levels_sorted = sorted(levels_norm, key=lambda x: priority.get(x, 0), reverse=True)
                    chosen = levels_sorted[0]

                evidence.update({
                    'interaction_level': chosen,
                    'ddinter_rows': int(len(rows)),
                    'matches': rows,
                    'chunk_ids': [r['chunk_id'] for r in rows],
                    'source_rows': [r['source_row'] for r in rows],
                })

            pairs.append(evidence)

    return pairs


def generate_response(user_message: str, language: str, resolved: List[Dict[str, Any]], retrievals: List[Dict[str, Any]]) -> str:
    """Build concise sections from verified medications and DDInter evidence only."""
    arabic = language in ('ar', 'arabic', 'mixed')
    meds = []
    for item in resolved:
        if item.get('verified'):
            generic = item.get('generic_name') or (item.get('active_ingredients') or [None])[0]
            input_name = item.get('input')
            name = input_name or generic
            if input_name and generic and normalize_text(input_name) != normalize_text(generic):
                name = f'{input_name} ({generic})'
            if name and name not in meds:
                meds.append(name)

    if language == 'arabizi':
        lines = ['💊 El adweya elly et3raf 3aleha']
        lines.extend(f'- {name}' for name in meds)
        if retrievals:
            lines.append('🔍 Natiget el drug interaction')
            for item in retrievals:
                if item.get('ddinter_rows') and item.get('interaction_level'):
                    lines.append(f"- Feh interaction ma3roofa ben {item['drug_a']} w {item['drug_b']} fe DDInter.")
                else:
                    lines.append(f"- Mafeesh interaction ma3roofa ben {item['drug_a']} w {item['drug_b']} fe data el DDInter el mota7a.")
            levels = [item.get('interaction_level') for item in retrievals if item.get('interaction_level')]
            if levels:
                lines.extend(['⚠️ Severity', f"- {', '.join(sorted(set(level.capitalize() for level in levels)))}"])
            lines.extend(['📌 El tafseel', '- El result da mabny bas 3ala records matched fe DDInter.', '📚 El evidence', '- Source: DDInter'])
            for item in retrievals:
                if item.get('chunk_ids'):
                    lines.append(f"- Record ID: {', '.join(item['chunk_ids'])}")
        if meds:
            lines.extend(['⚠️ Safety Note', '- Matwa2afsh wala te8ayar dawa prescribed men 8eer ma tes2al doctor aw pharmacist.'])
        return '\n'.join(lines)

    if arabic:
        lines = ['💊 الأدوية التي تم اكتشافها']
        lines.extend(f'- {name}' for name in meds)
        if retrievals:
            lines.append('🔍 نتيجة التداخل الدوائي')
            for item in retrievals:
                if item.get('ddinter_rows') and item.get('interaction_level'):
                    lines.append(f"- تم تسجيل تفاعل بين {item['drug_a']} و{item['drug_b']} في DDInter.")
                else:
                    lines.append(f"- لم يتم تحديد تفاعل بين {item['drug_a']} و{item['drug_b']} في قاعدة بيانات DDInter المتاحة.")
            levels = [item.get('interaction_level') for item in retrievals if item.get('interaction_level')]
            if levels:
                lines.extend(['⚠️ درجة الخطورة', f"- {', '.join(sorted(set(level.capitalize() for level in levels)))}"])
            lines.extend(['📌 التوضيح', '- هذه النتيجة مبنية على السجلات المطابقة في قاعدة بيانات DDInter فقط.', '📚 المصدر', '- DDInter'])
            for item in retrievals:
                if item.get('chunk_ids'):
                    lines.append(f"- السجل: {', '.join(item['chunk_ids'])}")
        if meds:
            lines.extend(['⚠️ ملاحظة السلامة', '- لا تغيّر أو توقف دواءً موصوفًا دون استشارة الطبيب أو الصيدلي.'])
        return '\n'.join(lines)

    lines = ['💊 Medications Detected']
    lines.extend(f'- {name}' for name in meds)
    if retrievals:
        lines.append('🔍 Interaction Result')
        for item in retrievals:
            if item.get('ddinter_rows') and item.get('interaction_level'):
                lines.append(f"- Interaction detected between {item['drug_a']} and {item['drug_b']} in DDInter.")
            else:
                lines.append(f"- No interaction was identified in the available DDInter knowledge base for {item['drug_a']} and {item['drug_b']}.")
        levels = [item.get('interaction_level') for item in retrievals if item.get('interaction_level')]
        if levels:
            lines.extend(['⚠️ Severity', f"- {', '.join(sorted(set(level.capitalize() for level in levels)))}"])
        lines.extend(['📌 Explanation', '- This result is based only on matching records in the DDInter database.', '📚 Evidence', '- Source: DDInter'])
        for item in retrievals:
            if item.get('chunk_ids'):
                lines.append(f"- Record ID: {', '.join(item['chunk_ids'])}")
    if meds:
        lines.extend(['⚠️ Safety Note', '- Do not stop or change a prescribed medication without consulting your doctor or pharmacist.'])
    return '\n'.join(lines)


def process_message(user_message: str, conversation_context: Dict[str, Any] = None, active_medications: List[str] = None) -> Dict[str, Any]:
    if conversation_context is None:
        conversation_context = {}
    if active_medications is None:
        active_medications = []

    language_style = detect_language_style(user_message)
    lang = detect_language(user_message)

    # handle pending clarification in context-aware mode
    pending = conversation_context.get('pending_clarification')
    # simple affirmative/negative token sets
    affirm = {'yes', 'yeah', 'yep', 'نعم', 'ايوه', 'أيوه', 'نعم', 'y'}
    negative = {'no', 'nah', 'لا', 'n'}
    user_norm = normalize_text(user_message)
    if pending and user_norm:
        # if user simply confirms
        if user_norm in affirm:
            # attempt to accept pending term as intended and re-resolve it
            suggestion = pending.get('suggestion') or {}
            span = suggestion.get('user_term') or pending.get('input') or pending.get('normalized_input')
            # re-resolve the pending short term
            new_res = resolve_drug(span, ddinter_csv_path=MASTER_INTERACTIONS_PATH)
            # mark as user_confirmed for traceability
            new_res['user_confirmed'] = True
            # clear pending in context
            conversation_context.pop('pending_clarification', None)
            # Build response using this confirmed term
            resolved = [new_res]
            verified_generics = []
            if new_res.get('verified'):
                for ingredient in (new_res.get('active_ingredients') or [new_res.get('generic_name')]):
                    if ingredient:
                        verified_generics.append(ingredient)
                
            all_generics = list(verified_generics)
            for m in active_medications:
                if m not in all_generics:
                    all_generics.append(m)
                    
            retrievals = retrieve_interactions(all_generics, resolved, new_medications=verified_generics)
            response = generate_response(user_message, language_style, resolved, retrievals)
            return {
                'language': lang,
                'language_style': language_style,
                'extracted': [span],
                'resolved': resolved,
                'verified_generics': verified_generics,
                'retrievals': retrievals,
                'response': response,
            }
        if user_norm in negative:
            # user rejected the suggestion
            conversation_context.pop('pending_clarification', None)
            res = {
                'language': lang,
                'language_style': language_style,
                'extracted': [],
                'resolved': [],
                'verified_generics': [],
                'retrievals': [],
                'response': (
                    'من فضلك اذكر اسم الدواء الصحيح.'
                    if language_style in ('arabic', 'mixed')
                    else 'Momken tekteb esm el dawa el sa7?' if language_style == 'arabizi'
                    else 'Please provide the correct medication name.'
                )
            }
            return res
    is_followup, followup_extracted = parse_medication_followup(user_message)
    if is_followup:
        extracted = followup_extracted
    else:
        extracted = extract_medications(user_message)
        
    resolved = resolve_medications(extracted)
    # Defensive filter: drop any resolved entry that equals the full user message
    # (prevents accidental full-sentence echoes from RxNav appearing to users)
    # Only drop if the message is longer than 2 words to allow single-word medication lookups.
    user_norm = normalize_text(user_message)
    resolved = [r for r in resolved if not (normalize_text(r.get('input')) == user_norm and len(user_norm.split()) > 2)]

    clarification = []
    verified_generics = []
    # Build a set of known safe generics (aliases + unique names) to avoid accepting API-only echo matches
    known_generics = set([v for v in SAFE_ALIASES.values()])
    try:
        for n in _load_unique_names():
            known_generics.add(normalize_text(n))
    except Exception:
        pass

    unique_resolved = []
    seen_generics = set()
    seen_unverified = set()

    for r in resolved:
        if r.get('verified'):
            ingredients = r.get('active_ingredients') or [r.get('generic_name')]
            gen = ingredients[0] if ingredients else None
            # safety check: if the API simply echoed the input as generic and that generic
            # is not in our known lists, treat as ambiguous (do not guess)
            if r.get('source') == 'rxnav' and gen and normalize_text(gen) == normalize_text(r.get('input')) and normalize_text(gen) not in known_generics:
                r['verified'] = False
                r['reason'] = 'Ambiguous RxNav echo match; treated as unverified by pipeline.'
                input_norm = normalize_text(r.get('input'))
                if input_norm not in seen_unverified:
                    seen_unverified.add(input_norm)
                    clarification.append(r)
                    unique_resolved.append(r)
                continue
            for ing in ingredients:
                if ing:
                    gen_norm = normalize_text(ing)
                    if gen_norm not in seen_generics:
                        seen_generics.add(gen_norm)
                        verified_generics.append(ing)
            unique_resolved.append(r)
        else:
            input_norm = normalize_text(r.get('input'))
            if input_norm not in seen_unverified:
                seen_unverified.add(input_norm)
                clarification.append(r)
                unique_resolved.append(r)
                
    resolved = unique_resolved

    all_generics = list(verified_generics)
    if active_medications and (is_followup or (len(extracted) == 1)):
        for m in active_medications:
            if m not in all_generics:
                all_generics.append(m)

    retrievals = retrieve_interactions(
        all_generics, 
        resolved, 
        new_medications=verified_generics if (active_medications and (is_followup or len(extracted) == 1)) else None
    )

    response = None
    if clarification:
        # ask for clarification about first ambiguous term (do not proceed until clarified)
        first = clarification[0]
        # store pending clarification in conversation context for follow-up
        conversation_context['pending_clarification'] = first
        if language_style in ('arabic', 'mixed'):
            suggestion = first.get('suggestion')
            if suggestion:
                response = f"لم أتمكن من تحديد {first.get('input')}. هل كنت تقصد {suggestion['user_term']} ({suggestion['generic_name']})؟\nلن أُجري تحليل التداخل حتى تؤكد اسم الدواء."
            else:
                response = f"لم أتمكن من تحديد {first.get('input')}. من فضلك وضح اسم الدواء.\nلن أُجري تحليل التداخل حتى تؤكد اسم الدواء."
        elif language_style == 'arabizi':
            suggestion = first.get('suggestion')
            if suggestion:
                response = f"Msh 3aref {first.get('input')}. 2asdak {suggestion['user_term']} ({suggestion['generic_name']})?\nMesh ha3mel interaction check 7atta t2aked esm el dawa."
            else:
                response = f"Msh 3aref {first.get('input')}. Momken twadda7 esm el dawa?\nMesh ha3mel interaction check 7atta t2aked esm el dawa."
        else:
            suggestion = first.get('suggestion')
            if suggestion:
                response = f"I couldn't identify '{first.get('input')}'. Did you mean {suggestion['user_term']} ({suggestion['generic_name']})?\nI won't run the interaction check until you confirm the medication name."
            else:
                response = f"I couldn't identify '{first.get('input')}'. Please clarify the medication name.\nI won't run the interaction check until you confirm the medication name."
    else:
        response = generate_response(user_message, language_style, resolved, retrievals)

    return {
        'language': lang,
        'language_style': language_style,
        'extracted': extracted,
        'resolved': resolved,
        'verified_generics': verified_generics,
        'retrievals': retrievals,
        'response': response,
    }

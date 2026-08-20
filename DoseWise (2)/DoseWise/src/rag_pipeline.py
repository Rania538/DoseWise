import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

from src.drug_resolver import normalize_text, resolve_drug, SAFE_ALIASES

ROOT = Path(__file__).resolve().parents[1]
MASTER_INTERACTIONS_PATH = ROOT / 'data' / 'processed' / 'master_interactions.csv'
UNIQUE_NAMES_PATH = ROOT / 'data' / 'processed' / 'unique_drug_names.csv'


def detect_language(text: str) -> str:
    if not text:
        return 'en'
    # Arabic script
    if re.search(r'[\u0600-\u06FF]', text):
        return 'ar'
    # Arabizi common digits used as phonetic markers (3,7,2)
    if re.search(r'\b[\w]*[37][\w]*\b', text) and re.search(r'[0-9]', text):
        return 'ar'
    return 'en'


def _load_unique_names() -> List[str]:
    if not UNIQUE_NAMES_PATH.exists():
        return []
    df = pd.read_csv(UNIQUE_NAMES_PATH, dtype=str)
    return df['normalized'].dropna().astype(str).tolist()


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
    alias_variants = {'بانادول': 'بنادول'}

    # check SAFE_ALIASES keys first (these are user-facing aliases/brands)
    for alias in SAFE_ALIASES.keys():
        pattern = r'\b' + re.escape(normalize_text(alias)) + r'\b'
        if re.search(pattern, norm):
            candidates.append(alias)
    for variant, alias in alias_variants.items():
        if re.search(r'\b' + re.escape(normalize_text(variant)) + r'\b', norm):
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
        'انا', 'باخد'
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

    df = pd.read_csv(MASTER_INTERACTIONS_PATH, dtype=str)
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
    # Build a safe, non-inventive answer from retrieved evidence
    intro_en = "According to the available interaction database,"
    intro_ar = "بحسب قاعدة البيانات المستخدمة،"

    if language == 'ar':
        lines = [intro_ar]
        for r in retrievals:
            if r['ddinter_rows'] > 0 and r['interaction_level']:
                lines.append(f"تم العثور على تفاعل بين {r['drug_a']} و{r['drug_b']} بدرجة {r['interaction_level'].capitalize()}.")
            else:
                lines.append(f"لم يتم تحديد تفاعل بين {r['drug_a']} و{r['drug_b']} في قاعدة البيانات المتاحة.")
        lines.append("لا تغيّر أو توقف أي دواء موصوف لك من نفسك، ويفضل مراجعة الطبيب أو الصيدلي.")
        return ' '.join(lines)

    # default English
    lines = [intro_en]
    for r in retrievals:
        if r['ddinter_rows'] > 0 and r['interaction_level']:
            lines.append(f"a {r['interaction_level'].capitalize()} interaction was identified between {r['drug_a']} and {r['drug_b']}.")
        else:
            lines.append(f"No interaction was identified in the available DDInter knowledge base for {r['drug_a']} and {r['drug_b']}.")

    lines.append("Do not stop or change a prescribed medication without consulting your doctor or pharmacist.")
    return ' '.join(lines)


def process_message(user_message: str, conversation_context: Dict[str, Any] = None, active_medications: List[str] = None) -> Dict[str, Any]:
    if conversation_context is None:
        conversation_context = {}
    if active_medications is None:
        active_medications = []

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
            span = pending.get('input') or pending.get('normalized_input')
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
                verified_generics.append(new_res.get('generic_name') or (new_res.get('active_ingredients') or [None])[0])
                
            all_generics = list(verified_generics)
            for m in active_medications:
                if m not in all_generics:
                    all_generics.append(m)
                    
            retrievals = retrieve_interactions(all_generics, resolved, new_medications=verified_generics)
            response = generate_response(user_message, lang, resolved, retrievals)
            return {
                'language': lang,
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
                'extracted': [],
                'resolved': [],
                'verified_generics': [],
                'retrievals': [],
                'response': 'Please provide the correct medication name.' if lang == 'en' else 'من فضلك اذكر اسم الدواء الصحيح.'
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
            gen = r.get('generic_name') or (r.get('active_ingredients') or [None])[0]
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
            if gen:
                gen_norm = normalize_text(gen)
                if gen_norm not in seen_generics:
                    seen_generics.add(gen_norm)
                    verified_generics.append(gen)
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
        if lang == 'ar':
            response = f"هل كنت تقصد {first.get('input')} ({first.get('normalized_input')})؟"
        else:
            response = f"Did you mean {first.get('input')}?"
    else:
        response = generate_response(user_message, lang, resolved, retrievals)

    return {
        'language': lang,
        'extracted': extracted,
        'resolved': resolved,
        'verified_generics': verified_generics,
        'retrievals': retrievals,
        'response': response,
    }

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


MAX_HISTORY_MESSAGES = 20
# Qwen3 handles Arabic, Arabizi/Franco, English, and mixed prompts well.
# GROQ_MODEL remains an escape hatch for deployments that use another model.
DEFAULT_MODEL = 'qwen/qwen3.8-27b'

OUT_OF_SCOPE_RESPONSE = (
    "Sorry, that's outside my scope. I'm DoseWise, a medication interaction assistant. "
    "I can help with medication-related questions, interactions, and medication safety."
)


def _load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _model_name() -> str:
    """Return the configured Qwen model used by every Groq call."""
    _load_environment()
    return os.getenv('GROQ_MODEL', DEFAULT_MODEL)


def _evidence_context(result: Dict[str, Any]) -> str:
    retrievals = result.get('retrievals', []) or []
    has_interaction = any(bool(item.get('ddinter_rows') and item.get('interaction_level')) for item in retrievals)
    severities = sorted(list(set(item['interaction_level'].capitalize() for item in retrievals if item.get('interaction_level'))))
    queried = [r.get('input') for r in result.get('resolved', []) if r.get('input')] or result.get('extracted', [])
    evidence = {
        'source': 'DDInter',
        'queried_medications': queried,
        'verified_active_ingredients': result.get('verified_generics', []),
        'interaction_found': has_interaction,
        'interaction_severity': severities,
        'ddinter_records': [
            {
                'drug_a': item.get('drug_a'),
                'drug_b': item.get('drug_b'),
                'interaction_level': item.get('interaction_level'),
                'chunk_ids': item.get('chunk_ids', []),
            }
            for item in retrievals
        ],
        # Backward-compatible keys
        'verified_medications': result.get('verified_generics', []),
        'ddinter_retrievals': retrievals,
    }
    return json.dumps(evidence, ensure_ascii=False, default=str)


def _clean_reply(reply: Optional[str]) -> Optional[str]:
    if not reply:
        return None
    cleaned = reply.strip()
    cleaned = re.sub(
        r'(?:<think>|(?<!<)think>).*?(?:</think>|$)',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r'</think>', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned or None


def _is_medication_result(result: Dict[str, Any]) -> bool:
    """Return True when the pipeline produced verified medications or retrievals."""
    return bool(result.get('verified_generics') or result.get('retrievals'))


def matches_language_style(reply: str, language: str) -> bool:
    """Reject model replies that visibly ignore the detected writing style."""
    if not reply:
        return False
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', reply))
    if language in ('arabic', 'mixed'):
        return has_arabic
    if language == 'arabizi':
        if has_arabic:
            return False
        arabizi_markers = re.findall(
            r'\b(?:ana|enta|enti|ynf3|m3|akhod|akhd|momken|ezay|ba5od|'
            r'ma3|mafeesh|fe|el|w|ben|dawa|adweya|ma3roofa|mota7a|'
            r'izayak|3amel|eh|msh|3aref|tmam|7abibi|wallahi|yb2a|keda|mfhmsh|leh|kwayes|katbly|doktor|tany|hom)\b',
            reply.casefold(),
        )
        # One marker can appear in an English sentence by coincidence; require
        # a distinctly Arabizi-shaped answer before accepting model output.
        return len(arabizi_markers) >= 2
    return not has_arabic


def has_pathological_repetition(text: Optional[str]) -> bool:
    """Detect pathological token or phrase repetition loops in generated output.

    Identifies consecutive repeated phrases (1 to 6 words) repeated 3 or more times,
    such as 'el 3awya el 3awya el 3awya' or 'word word word'.
    Does NOT flag natural double words (e.g. 'very very') or legitimate lists.
    """
    if not text:
        return False
    pattern = r'\b(\w+(?:[^\w\r\n]+\w+){0,5})\b(?:[\s,.;:!?\-]+\1\b){2,}'
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def validate_grounded_reply(reply: Optional[str], result: Dict[str, Any]) -> bool:
    """Verify that a generated response contains no unsupported medical claims.

    Rejects responses that:
    1. Claim medications are universally safe or risk-free when no interaction was found.
    2. Claim an interaction exists when DDInter found none.
    3. Assert a severity level not present in DDInter evidence.
    4. Assert unsupported contraindications or specific prescriptive dosage instructions.
    5. Fabricate DDInter citations or diagnoses in general health conversational mode.
    6. Contain pathological token or phrase repetition loops.
    """
    if not reply:
        return False

    if has_pathological_repetition(reply):
        return False

    reply_lower = reply.casefold()
    is_med = _is_medication_result(result)
    retrievals = result.get('retrievals', []) or []
    has_interaction = any(bool(item.get('ddinter_rows') and item.get('interaction_level')) for item in retrievals)
    allowed_severities = set(item['interaction_level'].casefold() for item in retrievals if item.get('interaction_level'))

    # Conversational mode (no recognized medications):
    if not is_med:
        # Must not fabricate DDInter citations or pretend DDInter verified general health advice
        if re.search(r'\bddinter\b', reply_lower):
            return False
        # Must not provide prescriptive clinical dosage instructions (e.g. "take 500mg", "take 2 pills")
        if re.search(r'\b(?:take|prescribe|dose\s+of)\s+\d+\s*(?:mg|ml|mcg|pills?|tablets?|capsules?)\b', reply_lower):
            return False
        # Must not make clinical diagnoses
        if re.search(r'\b(?:you\s+have\s+been\s+diagnosed\s+with|i\s+diagnose\s+you\s+with|my\s+diagnosis\s+is)\b', reply_lower):
            return False

    # Medication interaction mode:
    if is_med:
        # 1. Universal safety claim prohibition when no interaction was found
        if not has_interaction:
            unsafe_patterns = (
                r'\bcompletely\s+safe\b',
                r'\b100%\s+safe\b',
                r'\btotally\s+safe\b',
                r'\bguaranteed\s+safe\b',
                r'\bsafe\s+(?:to\s+take\s+together|together)\s+without\s+(?:any\s+)?risk\b',
                r'\bno\s+risk\s+at\s+all\b',
                r'\bthere\s+is\s+no\s+risk\b',
                r'\bآمن\s+تمام[اًا]\b',
                r'\bآمنة\s+تمام[اًا]\b',
                r'\bmafeesh\s+ay\s+risk\b',
                r'\bamen\s+tamaman\b',
            )
            if any(re.search(pat, reply_lower) for pat in unsafe_patterns):
                return False

            # 2. Invented interaction when DDInter found no interaction
            invented_patterns = (
                r'\binteraction\s+(?:detected|found|identified)\b',
                r'\bfeh\s+interaction\b',
                r'\bتم\s+تسجيل\s+تفاعل\b',
                r'\bيوجد\s+تداخل\b',
            )
            if any(re.search(pat, reply_lower) for pat in invented_patterns):
                return False

        # 3. Severity validation: reject any severity level not in allowed_severities
        all_severities = {'major', 'moderate', 'minor'}
        unallowed_severities = all_severities - allowed_severities
        for unallowed in unallowed_severities:
            pattern = rf'(?:severity|درجة\s+الخطورة|خطورة)[\s\:\-]+.*?\b{unallowed}\b'
            if re.search(pattern, reply_lower):
                return False

        # 4. Unsupported contraindications (DDInter provides interaction level, not formal contraindication)
        if re.search(r'\b(?:strictly\s+contraindicated|is\s+contraindicated|contraindication\s+exists)\b', reply_lower):
            return False

        # 5. Unsupported dosage advice in medication mode
        if re.search(r'\b(?:take|increase\s+dose\s+to|decrease\s+dose\s+to|adjust\s+dose\s+to)\s+\d+\s*(?:mg|ml|mcg|pills?|tablets?)\b', reply_lower):
            return False

    return True


_SCOPE_CLASSIFIER_PROMPT = (
    'You are an intent classifier for DoseWise, a medication interaction assistant. '
    'Determine whether the user message is COMPLETELY outside DoseWise scope. '
    'IN-SCOPE topics: medication questions, medication names, drug interactions, '
    'medication safety, dosing concerns, side effects, relevant general health guidance, '
    'and normal conversational greetings (hi, thanks, hello, how are you, etc.). '
    'OUT-OF-SCOPE topics: programming/code requests, unrelated famous people or history, '
    'jokes, general geography or trivia, sports, entertainment, politics, cooking, etc. '
    'Reply with exactly one word: YES (out of scope) or NO (in scope). '
    'No other text, no punctuation.'
)


def is_out_of_scope(message: str) -> bool:
    """Return True when the user message is entirely outside DoseWise domain.

    Uses the Groq LLM as a lightweight binary classifier (YES/NO, max 10 tokens).
    Falls back to False (in-scope) on any error so the pipeline stays available
    when Groq is unavailable or overloaded.
    """
    _load_environment()
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return False  # no Groq – cannot classify, let pipeline decide

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=_model_name(),
            messages=[
                {'role': 'system', 'content': _SCOPE_CLASSIFIER_PROMPT},
                {'role': 'user', 'content': message},
            ],
            temperature=0.0,
            max_tokens=500,  # generous budget so the thinking model can reason then answer
        )
        raw = completion.choices[0].message.content or ''
        cleaned = _clean_reply(raw) or ''
        # Accept YES anywhere in the cleaned reply (handles 'YES.', 'Yes', etc.)
        return bool(cleaned.strip().upper().startswith('YES'))
    except Exception as exc:
        print(f'Scope classifier unavailable: {exc}')
        return False  # fail open: let the normal pipeline handle it


def generate_reply(
    user_message: str,
    result: Dict[str, Any],
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """Return a grounded conversational reply, or None when Groq is unavailable.

    Uses a conversational system prompt for normal messages (no medications) and
    the medication-specialist prompt (with DDInter evidence) when the pipeline has
    returned verified medications or retrieval results.
    """
    _load_environment()
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return None

    try:
        from groq import Groq

        language = result.get('language_style', result.get('language', 'english'))
        # Keep the detected style explicit in the call boundary so it cannot
        # be lost when the result is passed through the conversational layer.
        detected_language = language

        if _is_medication_result(result):
            # Medication mode: grounded specialist prompt with DDInter evidence.
            system_prompt = (
                'You are Qwen powering DoseWise, a careful medication interaction assistant. '
                f"User's detected language/style: {detected_language}. Respond in this exact same style. Do not switch styles unless the user does. "
                'The detected language/style is authoritative: English means English; Arabic means standard Arabic; '
                'Arabizi means natural Latin-script Arabizi and never Arabic script; mixed means use the dominant style. '
                'Never default to English when the detected style is Arabic, Arabizi, or mixed. '
                'Arabizi digit mappings are Arabic letters: 2=hamza, 3=ain, 5=kha, 6=ta, 7=ha, 8=qaf/ghain, 9=sad. '
                'Recognize words such as izayak, 3amel eh, momken, msh 3aref, tmam, mfhmsh, 7abibi, wallahi, yb2a, and keda. '
                'For ambiguous one-word greetings, use Arabic script unless Arabizi was detected. '
                'Keep technical and academic terms such as math, ML, and programming in English inside Arabic replies. '
                'Examples: Arabic "ينفع اخد Panadol مع amoxicillin؟" -> "أقدر أساعدك في التحقق من التداخل بين Panadol و amoxicillin..."; '
                'Arabizi "ynf3 akhod Panadol m3 amoxicillin?" -> "Momken ageeblak natiget el interaction ben Panadol w amoxicillin..."; '
                'English "Can I take Panadol with amoxicillin?" -> "I can check the interaction between Panadol and amoxicillin...". '
                'Medication names, dosage units, and medical/scientific terms must remain in their verified English/RxNorm form and must NOT be translated or transliterated. '
                'The deterministic DoseWise pipeline has already identified and validated medications. '
                'GROUNDING CONTRACT: '
                'DDInter is the SOLE source of medical truth for medication interactions. '
                'Use ONLY the supplied structured DDInter evidence for all medication interaction claims. '
                'Never invent an interaction, severity, contraindication, symptom, dosage advice, or medical fact. '
                'Severity must ONLY be reported if present in interaction_severity; NEVER invent a severity level. '
                'If interaction_found is false (or DDInter has no matching row), clearly state that no interaction was identified in the available DDInter knowledge base. '
                'CRITICAL SAFETY BOUNDARY: You must NEVER claim or imply that the medications are "completely safe", "guaranteed safe together", or "have no risk" merely because no interaction was identified in DDInter. '
                'Do not guess or silently correct an unknown medication. '
                'Return concise sections with these labels: 💊 Medications Detected, 🔍 Interaction Result, '
                '⚠️ Severity, 📌 Explanation, 📚 Evidence, and ⚠️ Safety Note. Omit empty sections (for instance, omit ⚠️ Severity when interaction_severity is empty). '
                'Use only verified medications and retrieved DDInter records. The pipeline language is ' + str(language) + '. '
                'Here is the current structured evidence:\n' + _evidence_context(result)
            )
        else:
            # Conversational mode: friendly general assistant, no evidence block injected.
            system_prompt = (
                'You are Qwen powering DoseWise, a friendly and knowledgeable health assistant. '
                f"User's detected language/style: {detected_language}. Respond in this exact same style. Do not switch styles unless the user does. "
                'The detected language/style is authoritative: English means English; Arabic means standard Arabic; '
                'Arabizi means natural Latin-script Arabizi and never Arabic script; mixed means use the dominant style. '
                'Never default to English when the detected style is Arabic, Arabizi, or mixed. '
                'Arabizi digit mappings are Arabic letters: 2=hamza, 3=ain, 5=kha, 6=ta, 7=ha, 8=qaf/ghain, 9=sad. '
                'Recognize words such as izayak, 3amel eh, momken, msh 3aref, tmam, mfhmsh, 7abibi, wallahi, yb2a, and keda. '
                'For ambiguous one-word greetings, use Arabic script unless Arabizi was detected. '
                'For everyday greetings, small talk, or polite check-ins (such as "hello", "enta kwis?", "ezayak", "how are you"): '
                'Respond warmly, naturally, and concisely in 1-2 sentences in the user\'s detected style. Do not launch into lengthy disclaimers or repetitive speeches. Never repeat words or phrases. '
                'Examples: '
                '- Arabizi ("enta kwis?" / "ezayak"): "Ana kwayes elhamdolellah, shokran! A2dar asa3dak ezay fe el adweya ennaharda?" '
                '- Arabic ("عامل إيه؟" / "أهلاً"): "أنا بخير والحمد لله، شكراً لسؤالك! كيف أقدر أساعدك اليوم في أدويتك؟" '
                '- English ("how are you?" / "hello"): "I am doing well, thank you! How can I help you with your medications today?" '
                'Keep technical and academic terms such as math, ML, and programming in English inside Arabic replies. '
                'Medication names, dosage units, and medical/scientific terms must remain in their verified English/RxNorm form and must NOT be translated or transliterated. '
                'KNOWLEDGE BOUNDARIES: '
                'DoseWise specializes in checking medication interactions using data from the DDInter database. '
                'You can help with general health questions and normal conversations. '
                'For general health inquiries (such as common symptoms or wellness concepts): '
                '1. Provide concise, helpful general educational guidance only. '
                '2. NEVER claim or imply that general health answers come from DDInter records or verified DoseWise database data. Do NOT fabricate citations or record IDs. '
                '3. NEVER make clinical diagnoses, prescribe medications, or provide dosage advice. '
                '4. If symptoms or conditions are discussed, advise the user to consult a doctor or healthcare professional for medical evaluation. '
                '5. If asked specifically about medication interactions, let the user know they can ask you directly with the medication names and you will check the database. '
                'The conversation language/style is ' + str(language) + '.'
            )

        messages: List[Dict[str, str]] = [{'role': 'system', 'content': system_prompt}]
        if history:
            messages.extend(history[-MAX_HISTORY_MESSAGES:])
        messages.append({'role': 'user', 'content': user_message})

        client = Groq(api_key=api_key)
        call_kwargs: Dict[str, Any] = {
            'model': _model_name(),
            'messages': messages,
            'temperature': 0.3,
            'max_tokens': 500,
        }
        try:
            import inspect
            sig = inspect.signature(client.chat.completions.create)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if accepts_kwargs or 'frequency_penalty' in sig.parameters:
                call_kwargs['frequency_penalty'] = 0.5
                call_kwargs['presence_penalty'] = 0.3
        except Exception:
            pass

        completion = client.chat.completions.create(**call_kwargs)
        reply = _clean_reply(completion.choices[0].message.content)
        if not matches_language_style(reply, language):
            print(f'Qwen reply rejected: expected {language} response style')
            return None
        if has_pathological_repetition(reply):
            print('Qwen reply rejected: pathological repetition loop detected')
            return None
        if not validate_grounded_reply(reply, result):
            print('Qwen reply rejected: unsupported medical claim detected by grounding validator')
            return None
        return reply
    except Exception as exc:
        print(f'Groq reply unavailable: {exc}')
        return None
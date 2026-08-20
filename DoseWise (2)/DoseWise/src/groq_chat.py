import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


MAX_HISTORY_MESSAGES = 20
DEFAULT_MODEL = 'qwen/qwen3.6-27b'

OUT_OF_SCOPE_RESPONSE = (
    "Sorry, that's outside my scope. I'm DoseWise, a medication interaction assistant. "
    "I can help with medication-related questions, interactions, and medication safety."
)


def _load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _evidence_context(result: Dict[str, Any]) -> str:
    evidence = {
        'verified_medications': result.get('verified_generics', []),
        'ddinter_retrievals': result.get('retrievals', []),
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
            model=os.getenv('GROQ_MODEL', DEFAULT_MODEL),
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

        language = result.get('language', 'en')

        if _is_medication_result(result):
            # Medication mode: grounded specialist prompt with DDInter evidence.
            system_prompt = (
                'You are DoseWise, a careful medication interaction assistant. '
                'Reply naturally in the language and register of the latest user message. '
                'You may understand English, Arabic, and Arabizi and may mix them when the user does. '
                'The deterministic DoseWise pipeline has already identified and validated medications. '
                'Use only the supplied DDInter evidence for medication interaction claims. '
                'Never invent an interaction, severity, medication identity, or medical fact. '
                'If DDInter has no matching row, say that no interaction was identified in the available database; '
                'do not imply that this proves the combination is universally safe. '
                'Do not guess or silently correct an unknown medication. '
                'Keep answers concise, explain the relevant pair(s), and advise consulting a doctor or pharmacist '
                'before changing prescribed medication. The pipeline language is ' + str(language) + '. '
                'Here is the current structured evidence:\n' + _evidence_context(result)
            )
        else:
            # Conversational mode: friendly general assistant, no evidence block injected.
            system_prompt = (
                'You are DoseWise, a friendly and knowledgeable health assistant. '
                'Reply naturally in the language and register of the latest user message. '
                'You may understand English, Arabic, and Arabizi and may mix them when the user does. '
                'You can help with general health questions, explain what DoseWise does '
                '(it checks for medication interactions using evidence from the DDInter database), '
                'and have normal conversations. '
                'Never invent medical facts. If asked specifically about medication interactions, '
                'let the user know they can ask you directly and you will check the database. '
                'The conversation language is ' + str(language) + '.'
            )

        messages: List[Dict[str, str]] = [{'role': 'system', 'content': system_prompt}]
        if history:
            messages.extend(history[-MAX_HISTORY_MESSAGES:])
        messages.append({'role': 'user', 'content': user_message})

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv('GROQ_MODEL', DEFAULT_MODEL),
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        reply = completion.choices[0].message.content
        return _clean_reply(reply)
    except Exception as exc:
        print(f'Groq reply unavailable: {exc}')
        return None
from pathlib import Path
import json
import re
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

import src.rag_pipeline as rag
from src.groq_chat import MAX_HISTORY_MESSAGES, generate_reply
import importlib
FRONTEND_DIR = ROOT / 'frontend'

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
CORS(app)

# Simple in-memory session store for the current server process.
SESSIONS = {}

# Canonical out-of-scope refusal — shown for clearly unrelated questions.
OUT_OF_SCOPE_RESPONSE = (
    "Sorry, that's outside my scope. I'm DoseWise, a medication interaction assistant. "
    "I can help with medication interactions, medication safety, and related health questions."
)

# ---------------------------------------------------------------------------
# Deterministic out-of-scope classifier
# ---------------------------------------------------------------------------

# Regex patterns that clearly signal an out-of-scope request.
# Must NOT fire on medication or health questions — the escape hatches below
# (greeting / medication name / medication intent) take priority.
_OOS_PATTERNS = (
    # Person / entity information
    r'\bwho\s+(?:is|was|were|are)\b',
    r'\b(?:tell\s+me\s+about|any\s+info\s+about|info\s+about|what\s+do\s+you\s+know\s+about|give\s+me\s+info\s+(?:on|about))\b',
    # Programming / code generation
    r'\bwrite\s+(?:me\s+)?(?:a\s+|an\s+)?(?:python|java(?:script)?|ruby|c\+\+|golang|rust|php|typescript|program|script|code|function|algorithm|app(?:lication)?)\b',
    r'\b(?:code|program|script)\s+(?:for|that|to)\b',
    r'\bhow\s+(?:do\s+I|to)\s+(?:code|program|build|create)\b',
    # Geography / trivia
    r'\bcapital\s+of\b',
    r'\bpresident\s+of\b',
    r'\bprime\s+minister\s+of\b',
    r'\bpopulation\s+of\b',
    # Entertainment / humour
    r'\btell\s+(?:me\s+)?a\s+joke\b',
    r'\bmake\s+(?:me\s+)?(?:laugh|a\s+joke)\b',
    r'\bgive\s+(?:me\s+)?a\s+(?:joke|riddle)\b',
    # Cooking / recipes
    r'\brecipe\s+for\b',
    r'\bhow\s+to\s+(?:cook|bake|make)\s+[a-z]',
)

# Words/phrases at the START of a message that indicate a normal greeting.
_GREETING_START_RE = re.compile(
    r'^(?:hi|hello|hey|hiya|howdy|greetings|مرحبا|أهلا|السلام|هاي)',
    re.IGNORECASE,
)

# Standalone conversational tokens (entire message is one of these).
_STANDALONE_CONVERSATIONAL = frozenset([
    'hi', 'hello', 'hey', 'hiya', 'howdy',
    'thanks', 'thank you', 'thx', 'ty',
    'bye', 'goodbye', 'ok', 'okay', 'sure',
    'yes', 'no', 'yeah', 'nope', 'yep',
    'great', 'perfect', 'got it', 'understood',
    'how are you', 'how are you?',
    'مرحبا', 'أهلا', 'شكرا', 'شكراً',
])

# Medication domain keywords — presence means the message is in-scope.
_MED_DOMAIN_RE = re.compile(
    r'\b(?:medication|medicine|drug|pill|tablet|capsule|dose|dosage|'
    r'interaction|side\s+effect|pharmacist|pharmacy|prescription|'
    r'دواء|دواءين|علاج|صيدلي|جرعة|تفاعل)\b',
    re.IGNORECASE,
)


def _is_out_of_scope(message: str, pipeline_module) -> bool:
    """Return True when the message is clearly outside DoseWise's domain.

    Fully deterministic — no LLM call, no network dependency.

    Evaluation order (first match wins, False = in-scope):
    1. Message is a standalone conversational token  → in-scope
    2. Message starts with a greeting word (<= 10 words) → in-scope
    3. Message contains a recognised medication name   → in-scope
    4. Message contains a medication-domain keyword    → in-scope
    5. Message matches an OOS regex pattern            → out-of-scope
    6. Default                                         → in-scope
    """
    text = message.strip()
    lower = text.casefold()

    # 1. Standalone conversational token
    if lower.rstrip('!?,. ') in _STANDALONE_CONVERSATIONAL:
        return False

    # 2. Starts with a greeting word and is short (allows "Hi, my name is Ranya")
    if _GREETING_START_RE.match(text) and len(text.split()) <= 10:
        return False

    # 3. Known medication name present
    if _contains_recognized_medication(message, pipeline_module):
        return False

    # 4. Medication-domain keyword present
    if _MED_DOMAIN_RE.search(lower):
        return False

    # 5. Out-of-scope pattern
    return any(re.search(pattern, lower) for pattern in _OOS_PATTERNS)


def _session_context(session_id):
    if not session_id:
        return {}, None
    context = SESSIONS.setdefault(session_id, {'history': [], 'active_medications': []})
    context.setdefault('history', [])
    context.setdefault('active_medications', [])
    return context, session_id


def _is_medication_query(message, result):
    if any(item.get('verified') for item in result.get('resolved', [])):
        return True

    text = message.casefold()
    intent_terms = (
        r'\btake\b', r'\btaking\b', r'\btook\b', r'\bprescribed\b',
        r'\bmedicine\b', r'\bmedication\b', r'\bdrug\b', r'\bpill\b',
        r'\btablet\b', r'\bdose\b', r'\binteraction\b', r'\bpharmac\w*\b',
        r'\bاخد\b', r'\bآخد\b', r'\bباخد\b', r'\bدواء\b', r'\bدواءين\b',
        r'\bتفاعل\b', r'\bجرعة\b', r'\bصيدلي\b', r'\bكتبه?لي\b',
    )
    return any(re.search(term, text) for term in intent_terms)


def _contains_recognized_medication(message, pipeline_module):
    normalized = pipeline_module.normalize_text(message)
    names = list(pipeline_module.SAFE_ALIASES.keys())
    names.extend(pipeline_module._load_unique_names())
    names.append('بانادول')
    for name in names:
        candidate = pipeline_module.normalize_text(name)
        if candidate and re.search(
            r'(?<![\w])' + re.escape(candidate) + r'(?![\w])', normalized
        ):
            return True
    return False


def _has_safety_intent(message):
    text = message.casefold()
    safety_terms = (
        r'\boverdose\b', r'\btoo many\b', r'\btoo much\b',
        r'\bextra dose\b', r'\bdouble\s+(?:\w+\s+){0,2}dose\b', r'\baccidentally took\b',
        r'\bintentionally took\b', r'\bemergency\b', r'\bpoison\b',
        r'\bجرعة زائدة\b', r'\bاخدت كتير\b', r'\bبالغلط\b'
    )
    
    if re.search(r'\b([2-9]|[1-9][0-9]+)\s*(?:[A-Za-z\u0600-\u06FF0-9-]+\s*){0,3}(tablets?|pills?|capsules?|حبات|برشامات|اقراص|قرص)\b', text):
        return True
        
    return any(re.search(term, text) for term in safety_terms)


def _has_medication_intent(message):
    text = message.casefold()
    intent_terms = (
        r'\btake\b', r'\btaking\b', r'\btook\b', r'\bprescribed\b',
        r'\bmedicine\b', r'\bmedication\b', r'\bdrug\b', r'\bpill\b',
        r'\btablet\b', r'\bdose\b', r'\binteraction\b', r'\bpharmac\w*\b',
        r'\bsafe\b', r'\bside effects?\b',
        r'\bاخد\b', r'\bآخد\b', r'\bباخد\b', r'\bدواء\b', r'\bدواءين\b',
        r'\bتفاعل\b', r'\bجرعة\b', r'\bصيدلي\b', r'\bكتبه?لي\b',
    )
    return any(re.search(term, text) for term in intent_terms)


def _empty_conversation_result(language):
    return {
        'language': language,
        'extracted': [],
        'resolved': [],
        'verified_generics': [],
        'retrievals': [],
        'response': None,
    }


@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json(force=True)
        message = data.get('message', '') if isinstance(data, dict) else ''
        if not message or not str(message).strip():
            return jsonify({'error': 'Empty message'}), 400

        session_id = data.get('session_id') if isinstance(data, dict) else None
        conversation_context, session_key = _session_context(session_id)
        try:
            mod = importlib.import_module('src.rag_pipeline')
            mod = importlib.reload(mod)
        except Exception:
            print('DEBUG: failed to import/reload src.rag_pipeline, using existing rag')
            mod = rag

        # --- Safety Intent Check ---
        if _has_safety_intent(message):
            response_text = "This sounds like a potential medical emergency or overdose. Do NOT take any more medication. Please immediately contact your local poison control center or seek urgent medical help."
            resp = {
                'message': message,
                'language': mod.detect_language(message),
                'extracted': [],
                'resolved': [],
                'verified_generics': [],
                'retrievals': [],
                'response': response_text,
                'needs_clarification': False,
                'groq_used': False,
            }
            return jsonify(resp)

        # --- Out-of-scope Intent Check ---
        # Runs after safety (always highest priority) but before the medication
        # pipeline.  Fully deterministic — requires no Groq API call.
        if _is_out_of_scope(message, mod):
            resp = {
                'message': message,
                'language': mod.detect_language(message),
                'extracted': [],
                'resolved': [],
                'verified_generics': [],
                'retrievals': [],
                'response': OUT_OF_SCOPE_RESPONSE,
                'needs_clarification': False,
                'groq_used': False,
                'out_of_scope': True,
            }
            return jsonify(resp)

        # --- Determine if this message should enter the medication pipeline ---

        # 1) Check if this is a medication follow-up (e.g. "what about simvastatin?")
        is_followup, _ = mod.parse_medication_followup(message)
        has_active_meds = bool(conversation_context.get('active_medications'))

        # 2) Check if the message contains a recognized medication name or intent
        medication_message = (
            _contains_recognized_medication(message, mod)
            or _has_medication_intent(message)
        )

        # 3) A follow-up with active medications should always enter the pipeline
        if is_followup and has_active_meds:
            medication_message = True

        if not medication_message:
            result = _empty_conversation_result(mod.detect_language(message))
            needs_clarification = False
        else:
            # Keep the existing medication pipeline and session aggregation intact.
            try:
                active_meds = conversation_context.get('active_medications', [])
                # If it's a fresh query (not a follow-up), wipe the previous medications.
                if not is_followup:
                    active_meds = []
                    conversation_context['active_medications'] = []
                
                result = mod.process_message(
                    message,
                    conversation_context=conversation_context,
                    active_medications=active_meds
                )
                result['language'] = mod.detect_language(message)

                if session_key:
                    active = conversation_context.get('active_medications', [])
                    for medication in result.get('verified_generics', []):
                        if medication not in active:
                            active.append(medication)
            except Exception:
                print('DEBUG: failed to process medication pipeline')
                traceback.print_exc()
                result = rag.process_message(message, conversation_context=conversation_context)
                result['language'] = rag.detect_language(message)

            needs_clarification = (
                any(not r.get('verified') for r in result.get('resolved', []))
                and _is_medication_query(message, result)
            )

        groq_response = None
        if not needs_clarification:
            session_history = conversation_context.get('history', []) if session_key else []
            groq_response = generate_reply(message, result, session_history)
        response = groq_response or result.get('response')
        if response is None and not medication_message:
            response = 'I am here to help. What would you like to know?'

        if session_key:
            history = conversation_context.setdefault('history', [])
            history.append({'role': 'user', 'content': message})
            history.append({'role': 'assistant', 'content': response})
            del history[:-MAX_HISTORY_MESSAGES]

        resp = {
            'message': message,
            'language': result.get('language'),
            'extracted': result.get('extracted'),
            'resolved': result.get('resolved'),
            'verified_generics': result.get('verified_generics'),
            'retrievals': result.get('retrievals'),
            'response': response,
            'needs_clarification': needs_clarification,
            'groq_used': bool(groq_response),
        }
        return jsonify(resp)

    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/session', methods=['DELETE'])
def delete_session():
    """Delete a session's state. Called by the frontend on Clear."""
    try:
        data = request.get_json(force=True) if request.data else {}
        session_id = data.get('session_id') if isinstance(data, dict) else None
        if session_id and session_id in SESSIONS:
            del SESSIONS[session_id]
        return jsonify({'status': 'ok'})
    except Exception:
        return jsonify({'status': 'ok'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    # Serve static frontend files
    if path == '' or not (FRONTEND_DIR / path).exists():
        return send_from_directory(str(FRONTEND_DIR), 'index.html')
    return send_from_directory(str(FRONTEND_DIR), path)


if __name__ == '__main__':
    print('Starting DoseWise backend at http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000)

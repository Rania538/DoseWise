from pathlib import Path
import json
import logging
import os
import re
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

import src.rag_pipeline as rag
from src.groq_chat import MAX_HISTORY_MESSAGES, generate_reply, matches_language_style
FRONTEND_DIR = ROOT / 'frontend'

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024  # 100 KB max request body

DEFAULT_ALLOWED_ORIGINS = ["http://localhost:5000", "http://127.0.0.1:5000"]


def _get_allowed_origins():
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if raw.strip():
        parsed = [origin.strip() for origin in raw.split(",") if origin.strip()]
        # Do not weaken CORS by allowing wildcard '*'
        filtered = [origin for origin in parsed if origin != "*"]
        if filtered:
            return filtered
    return list(DEFAULT_ALLOWED_ORIGINS)


ALLOWED_ORIGINS = _get_allowed_origins()
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

import time
import secrets

# Session and security configurations
SESSIONS = {}
SESSION_TTL = 3600  # 1 hour
MAX_TOTAL_SESSIONS = 5000  # Max active sessions in memory
MAX_SESSION_MESSAGES = 100  # Max messages per session
MAX_MESSAGE_LENGTH = 4096  # Max characters per user message
MAX_PAYLOAD_BYTES = 64 * 1024  # 64 KB maximum JSON payload

CLIENT_COOKIE_NAME = 'dosewise_client_id'


def _is_cookie_secure() -> bool:
    """Determine whether cookies should be marked Secure.
    Supports explicit env setting, production environment flags, or HTTPS requests.
    Defaults to False for local HTTP development and automated tests.
    """
    env_flag = os.getenv("SECURE_COOKIES", "").strip().lower()
    if env_flag in ("1", "true", "yes"):
        return True
    if env_flag in ("0", "false", "no"):
        return False
    if os.getenv("DOSEWISE_ENV", "").strip().lower() == "production":
        return True
    try:
        if request.is_secure or request.headers.get("X-Forwarded-Proto") == "https":
            return True
    except RuntimeError:
        pass
    return False


def _set_client_cookie(response_obj, client_id):
    response_obj.set_cookie(
        CLIENT_COOKIE_NAME,
        client_id,
        httponly=True,
        samesite='Lax',
        secure=_is_cookie_secure(),
    )


class InMemoryRateLimiter:
    """Sliding-window in-memory rate limiter per client key (IP or client token)."""
    def __init__(self, max_requests=300, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        timestamps = self.requests.setdefault(key, [])
        cutoff = now - self.window_seconds
        valid = [t for t in timestamps if t > cutoff]
        self.requests[key] = valid
        if len(valid) >= self.max_requests:
            return False
        valid.append(now)
        return True

    def reset(self):
        self.requests.clear()


RATE_LIMITER = InMemoryRateLimiter(max_requests=300, window_seconds=60)


def _reset_rate_limits():
    """Helper for testing to reset rate limit counters."""
    RATE_LIMITER.reset()


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Request payload exceeds maximum allowed size.'}), 413


# Canonical out-of-scope refusal — shown for clearly unrelated questions.
OUT_OF_SCOPE_RESPONSE = (
    "Sorry, that's outside my scope. I'm DoseWise, a medication interaction assistant. "
    "I can help with medication interactions, medication safety, and related health questions."
)


def _localized_guardrail_response(language_style, kind):
    responses = {
        'safety': {
            'arabic': 'يبدو أن هذه حالة طبية طارئة أو جرعة زائدة. لا تتناول أي دواء إضافي. اتصل فورًا بمركز مكافحة السموم المحلي أو اطلب المساعدة الطبية العاجلة.',
            'arabizi': 'Da momken ykoon 7ala tebya ta2reya aw overdose. Matakhodsh ay dawa zyada. Et3amel foran ma3 markaz mokaf7et el somom aw otlob mosa3ada tebya 3agela.',
            'mixed': 'يبدو أن هذه حالة طبية طارئة أو overdose. لا تتناول أي دواء إضافي، واتصل فورًا بمركز مكافحة السموم المحلي أو اطلب urgent medical help.',
            'english': 'This sounds like a potential medical emergency or overdose. Do NOT take any more medication. Please immediately contact your local poison control center or seek urgent medical help.',
        },
        'out_of_scope': {
            'arabic': 'عذرًا، هذا خارج نطاقي. أنا DoseWise، مساعد للتداخلات الدوائية. أستطيع مساعدتك في أسئلة الأدوية والتداخلات وسلامة استخدامها.',
            'arabizi': 'Asf, da bara ne2aty. Ana DoseWise, mosa3ed le taf3olat el adweya. A2dar asa3dak fe as2elet el adweya w salamet estekhdamha.',
            'mixed': 'عذرًا، هذا outside my scope. أنا DoseWise، وأستطيع مساعدتك في medication interactions و medication safety.',
            'english': OUT_OF_SCOPE_RESPONSE,
        },
    }
    return responses[kind].get(language_style, responses[kind]['english'])

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


class SessionSecurityError(Exception):
    def __init__(self, message, status_code=403):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _get_or_create_client_id():
    """Retrieve or securely generate a client ownership identifier.
    Checks:
    1. HTTP header 'X-Client-ID' or 'X-Session-Token'
    2. Cookie 'dosewise_client_id'
    3. JSON payload 'client_id'
    Falls back to generating a cryptographically secure token.
    """
    client_id = request.headers.get('X-Client-ID') or request.headers.get('X-Session-Token')
    if client_id and str(client_id).strip():
        return str(client_id).strip(), False

    cookie_val = request.cookies.get(CLIENT_COOKIE_NAME)
    if cookie_val and str(cookie_val).strip():
        return str(cookie_val).strip(), False

    # Check if request has JSON with client_id
    if request.is_json and request.content_length and request.content_length <= MAX_PAYLOAD_BYTES:
        try:
            data = request.get_json(silent=True)
            if isinstance(data, dict) and data.get('client_id'):
                return str(data['client_id']).strip(), False
        except Exception:
            pass

    return secrets.token_urlsafe(32), True


def _session_context(session_id, client_id=None):
    now = time.time()
    # 1. Cleanup expired sessions
    expired = [k for k, v in SESSIONS.items() if now - v.get('last_accessed', 0) > SESSION_TTL]
    for k in expired:
        del SESSIONS[k]

    if not client_id:
        client_id = secrets.token_urlsafe(32)

    # If no session_id provided, generate an unpredictable identifier
    if not session_id or not str(session_id).strip():
        session_id = secrets.token_urlsafe(32)

    session_id = str(session_id).strip()

    # 2. Existing session: enforce ownership and limits
    if session_id in SESSIONS:
        session = SESSIONS[session_id]
        owner = session.get('owner_id')
        if owner and client_id and owner != client_id:
            raise SessionSecurityError("Unauthorized: Session belongs to another client", 403)

        # Enforce message count limit per session
        msg_count = session.get('message_count', 0)
        if msg_count >= MAX_SESSION_MESSAGES:
            raise SessionSecurityError(
                f"Session message limit exceeded ({MAX_SESSION_MESSAGES} messages). Please start a new session.",
                429
            )

        session['message_count'] = msg_count + 1
        session['last_accessed'] = now
        session.setdefault('history', [])
        session.setdefault('active_medications', [])
        return session, session_id

    # 3. New session: check capacity limit before creating
    if len(SESSIONS) >= MAX_TOTAL_SESSIONS:
        raise SessionSecurityError("Server session capacity reached. Please try again later.", 503)

    context = {
        'history': [],
        'active_medications': [],
        'last_accessed': now,
        'created_at': now,
        'owner_id': client_id,
        'message_count': 1,
    }
    SESSIONS[session_id] = context
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


def _is_educational_overdose_query(text: str) -> bool:
    """Return True when the message is clearly asking ABOUT overdose / excessive dose
    educationally, with no accompanying ingestion, self-harm, or active dose-taking context.

    Examples that return True:
        "What is overdose?"  "What does overdose mean?"
        "What are overdose symptoms?"  "Can overdose be dangerous?"
        "يعني ايه overdose؟"  "هو ايه ال overdose؟"
        "ايه أعراض الجرعة الزايدة؟"  "يعني ايه جرعة زائدة؟"
        "ya3ny eh overdose?"
    """
    # Educational question openers in English, Arabic and Franco-Arabic
    edu_question_re = re.compile(
        r'(?:'
        r'\b(?:what\s+(?:is|are|does|do|can)|how\s+(?:do|does|can|is|are)|how\s+to|'
        r'can\s+(?:you\s+)?(?:explain|describe|tell\s+me\s+about|clarify)|'
        r'can\s+(?:an?\s+)?overdose|could\s+(?:an?\s+)?overdose|is\s+(?:an?\s+)?overdose|'
        r'can\s+(?:someone|anybody|anyone|a\s+person|you|i|we)\s+overdose|'
        r'explain|define|meaning\s+of|definition\s+of|tell\s+me\s+about|info\s+about|information\s+about|'
        r'symptoms?\s+of\s+(?:an?\s+)?(?:overdose|the\s+overdose)|signs?\s+of\s+(?:an?\s+)?(?:overdose|the\s+overdose)|'
        r'risks?\s+of\s+(?:an?\s+)?(?:overdose|the\s+overdose)|'
        r'overdose\s+(?:symptoms?|signs?|risks?|effects?|definition|meaning))\b'
        r'|'
        # Arabic/Franco question words: يعني ايه / هو ايه / ما هو / ايه
        r'(?:يعني\s+(?:ايه|إيه|ما)|هو\s+(?:ايه|إيه|ما)|ما\s+(?:هو|هي|يعني|أعراض|اعراض|مخاطر|معنى)|'
        r'ايه\s+(?:ال|هو|هي|معنى|أعراض|اعراض|مخاطر)?|'
        r'أعراض\s+(?:الجرعة|overdose)|اعراض\s+(?:الجرعة|overdose)|'
        r'مخاطر\s+(?:الجرعة|overdose)|'
        r'ya3ny\s+eh|yani\s+eah|ma\s+huwa|eh\s+el|eh\s+howa|eh\s+heya|eh\s+a3rad|a3rad\s+el)'
        r')',
        re.IGNORECASE,
    )

    # Ingestion / action signals that override the educational classification
    ingestion_re = re.compile(
        r'(?:'
        r'\b(?:i\s+(?:took|take|taken|have\s+taken|just\s+took|already\s+took|swallowed|ingested|drank)|'
        r'(?:i\s+think\s+i|think\s+i)(?:\'ve|\s+have)?\s+overdosed?|'
        r'i\s+overdosed?|'
        r'(?:took|taken|ingested|swallowed)\s+(?:\w+\s+){0,4}(?:too\s+many|too\s+much|extra|more\s+than|\d+|two|three|four|five|six)|'
        r'overdosed?\s+(?:on|myself)|want\s+to\s+overdose|going\s+to\s+overdose|plan\s+to\s+overdose|trying\s+to\s+overdose)\b'
        r'|'
        r'(?:\u0627\u062e\u062f\u062a|\u062e\u062f\u062a|\u0623\u062e\u0630\u062a|\u0627\u062e\u0630\u062a|\u0627\u062e\u062f|\u062d\u0627\u0633\u0633|\u062d\u0627\u0633\u0647|\u0628\u0644\u0639\u062a|\u0634\u0631\u0628\u062a|\u062a\u0646\u0627\u0648\u0644\u062a)'
        r'|'
        r'\b(?:akhat|akht|5adt|khadt|akhadt)\b'
        r')',
        re.IGNORECASE,
    )

    # Self-harm signals immediately override educational classification
    self_harm_re = re.compile(
        r'(?:'
        r'\b(?:die|kill|hurt|harm|suicide|end\s+my\s+life|end\s+it\s+all)\b|'
        r'(?:\u0627\u0646\u062a\u062d\u0627\u0631|\u0627\u0646\u062a\u062d\u0631|\u0623\u0645\u0648\u062a|\u0627\u0645\u0648\u062a|\u0623\u0624\u0630\u064a|\u0623\u0622\u0630\u064a|\u0622\u0630\u0649|\u0627\u0630\u064a)|'
        r'\b(?:a2zy|a2zi|amot|amoot|enta7er)\b'
        r')',
        re.IGNORECASE,
    )

    has_edu_signal = bool(edu_question_re.search(text))
    has_ingestion_signal = bool(ingestion_re.search(text))
    has_self_harm_signal = bool(self_harm_re.search(text))

    return has_edu_signal and not has_ingestion_signal and not has_self_harm_signal


def _has_safety_intent(message):
    text = message.casefold()

    # ---------------------------------------------------------------------------
    # 1. Self-harm / suicide intent — always safety regardless of context
    # ---------------------------------------------------------------------------
    self_harm_terms = (
        # English — self-harm / suicide
        r'\bsuicide\b',
        r'\bkill\s+(?:my)?self\b',
        r'\bwant\s+to\s+(?:die|hurt\s+(?:my)?self|harm\s+(?:my)?self|kill\s+(?:my)?self|end\s+my\s+life)\b',
        r'\bend\s+it\s+all\b',
        r'\bend\s+my\s+life\b',
        r'\bhurt\s+(?:my)?self\b',
        r'\bharm\s+(?:my)?self\b',
        r'\b(?:don\'?t\s+want\s+to\s+live|do\s+not\s+want\s+to\s+live)\b',
        r'\b(?:want\s+to|plan\s+to|going\s+to|trying\s+to)\s+overdose\b',
        # Arabic — self-harm / suicide
        r'\u0627\u0646\u062a\u062d\u0627\u0631\b',                            # انتحار
        r'\u0627\u0646\u062a\u062d\u0631\b',                                  # انتحر
        r'(?:\u0639\u0627\u064a\u0632|\u0639\u0627\u064a\u0632\u0629|\u0628\u062f\u064a)\s+\u0623?\u0627?\u0645\u0648\u062a\b',  # عايز/عايزة/بدي اموت
        r'(?:\u0639\u0634\u0627\u0646|\u0644\u0643\u064a|\u062d\u062a\u0649)\s+\u0623?\u0627?\u0645\u0648\u062a\b',               # عشان اموت
        r'(?:\u0639\u0634\u0627\u0646|\u0644\u0643\u064a)\s+\u0627\u0646\u062a\u062d\u0627\u0631\b',                               # عشان انتحار
        r'(?:\u0639\u0634\u0627\u0646|\u0644\u0643\u064a)\s+\u0627\u0646\u062a\u062d\u0631\b',                                      # عشان انتحر
        r'[\u0623\u0625\u0622\u0627\u0624\u0621\u0626]{1,2}\u0630[\u064a\u0649]\s+\u0646\u0641\u0633[\u064a\u0649]\b',            # أؤذي / أأذي / آذي / اذي نفسي
        r'(?:\u0639\u0634\u0627\u0646|\u0644\u0643\u064a)\s+[\u0623\u0625\u0622\u0627\u0624\u0621\u0626]{1,2}\u0630[\u064a\u0649]\b',  # عشان اذي / أأذي
        r'(?:\u0648\u062e\u062f\u062a|\u0648\u0627\u062e\u062f\u062a)\s+\u0627\u0644\u062d\u0628\u0648\u0628\b',                   # وخدت الحبوب
        # Franco-Arabic
        r'\b(?:3ayz|3ayza|ayza|3awez)\s+(?:amot|amoot)\b',
        r'\b(?:a2zy|a2zi)\s+nafs(?:y|i)\b',
        r'\b(?:enta7er|ente7ar|enta7ar)\b',
    )
    if any(re.search(term, text) for term in self_harm_terms):
        return True

    # ---------------------------------------------------------------------------
    # 2. Pill/tablet/dose quantity pattern — possible real ingestion / overdose
    #    e.g. "I took 5 Panadol tablets", "اخدت 6 حبات", "I took 2 Panadol"
    # ---------------------------------------------------------------------------
    # Multi-unit quantity (tablets, pills, capsules, حبات, برشامات, اقراص)
    if re.search(
        r'\b([2-9]|[1-9][0-9]+)\s*(?:[A-Za-z\u0600-\u06FF0-9-]+\s*){0,3}'
        r'(tablets?|pills?|capsules?|\u062d\u0628\u0627\u062a|\u0628\u0631\u0634\u0627\u0645\u0627\u062a|\u0627\u0642\u0631\u0627\u0635|\u0642\u0631\u0635)\b',
        text,
    ):
        return True

    # Ingestion verb + quantity + medication (e.g. "I took 2 Panadol", "I took 6 pills")
    if re.search(
        r'\b(?:took|taken|swallowed|ingested)\s+(?:[2-9]|[1-9][0-9]+|two|three|four|five|six|seven|eight|nine|ten)\s+'
        r'(?:[A-Za-z\u0600-\u06FF0-9-]+)\b',
        text,
    ):
        return True

    # Arabic ingestion verb + quantity (e.g. "اخدت 6 حبات", "اخدت 6 حبات بالغلط", "اخدت 2 بانادول")
    if re.search(
        r'(?:\u0627\u062e\u062f\u062a|\u062e\u062f\u062a|\u0623\u062e\u0630\u062a|\u0627\u062e\u0630\u062a)\s+'
        r'(?:[2-9]|[1-9][0-9]+|\u0627\u062a\u0646\u064a\u0646)\s*',
        text,
    ):
        return True

    # Franco-Arabic ingestion + quantity (e.g. "akht 2 doses", "akhat 6 pills")
    if re.search(
        r'\b(?:akhat|akht|5adt|khadt|akhadt)\s+(?:[2-9]|[1-9][0-9]+|2)\s+'
        r'(?:doses?|pills?|tablets?|capsules?|panadol|7abat)\b',
        text,
    ):
        return True

    # ---------------------------------------------------------------------------
    # 3. Active ingestion / dosing errors
    # ---------------------------------------------------------------------------
    active_dosing_errors = (
        # English
        r'\btoo many\b',
        r'\btoo much\b',
        r'\bextra dose\b',
        r'\bdouble\s+(?:\w+\s+){0,2}dose\b',
        r'\baccidentally took\b',
        r'\bintentionally took\b',
        r'\bemergency\b',
        r'\bpoison\b',
        r'\btook\s+(?:.*?\s+)?by\s+mistake\b',
        r'\btook\s+(?:my\s+)?(?:medicine|medication|pills?|tablets?|capsules?|doses?)\s+(?:twice|again|two\s+times|multiple\s+times)\b',
        r'\btook\s+(?:\d+|two|three|multiple)\s+doses?\b',
        r'\b(?:think\s+i|i\s+think\s+i(?:\'ve)?|i)\s+overdosed?\b',
        r'\boverdosed\b',
        # Arabic
        r'\u0628\u0627\u0644\u063a\u0644\u0637\b',                        # بالغلط
        r'\u062c\u0631\u0639\u062a\u064a\u0646\b',                        # جرعتين
        r'(?:\u0627\u062e\u062f\u062a|\u062e\u062f\u062a|\u0623\u062e\u0630\u062a)\s+(?:\u0627\u0644\u062f\u0648\u0627\s+)?(?:\u0645\u0631\u062a\u064a\u0646|\u0632\u064a\u0627\u062f\u0629|\u0643\u062a\u064a\u0631)', # اخدت الدوا مرتين / زيادة / كتير
        r'(?:\u0627\u062e\u062f\u062a|\u062e\u062f\u062a|\u0623\u062e\u0630\u062a).*?\u062c\u0631\u0639\u0629\s+(?:\u0632\u064a\u0627\u062f\u0629|\u0632\u0627\u0626\u062f\u0629|\u0632\u0627\u064a\u062f\u0629)',  # اخدت جرعة زيادة
        r'\u062d\u0627\u0633\u0633.*?\u062c\u0631\u0639\u0629\s+(?:\u0632\u064a\u0627\u062f\u0629|\u0632\u0627\u0626\u062f\u0629|\u0632\u0627\u064a\u062f\u0629)',  # حاسس بجرعة زيادة / حاسس اني اخدت جرعة زيادة
        r'\u062d\u0627\u0633\u0633\s*(?:\u0627\u0646\u064a|\u0625\u0646\u064a)?\s*\u0627\u062e\u062f\u062a',  # حاسس إني اخدت
        # Franco-Arabic
        r'\b(?:akhat|akht|5adt|khadt)\s+(?:dosein|dosen|dawa\s+zyada|zyada|maraten)\b',
        r'\b(?:akhat|akht|5adt|khadt)\s+.*?belghalat\b',
        r'\b(?:dosein|dosen)\s+belghalat\b',
    )
    if any(re.search(term, text) for term in active_dosing_errors):
        return True

    # ---------------------------------------------------------------------------
    # 4. "overdose" / "جرعة زائدة" — context-aware: educational vs real emergency
    # ---------------------------------------------------------------------------
    has_overdose_term = bool(
        re.search(r'\boverdose[sd]?\b', text)
        or re.search(r'\u062c\u0631\u0639\u0629\s+(?:\u0632\u0627\u0626\u062f\u0629|\u0632\u064a\u0627\u062f\u0629|\u0632\u0627\u064a\u062f\u0629)\b', text)
    )

    if has_overdose_term:
        # Educational inquiry -> NOT an emergency
        if _is_educational_overdose_query(text):
            return False

        # Any overdose mention with ingestion/action context -> safety
        if re.search(
            r'\b(?:took|take|taken|i\s+think|i\s+may\s+have|may\s+have|might\s+have|swallowed|drank|ingested)\b'
            r'|(?:\u0627\u062e\u062f\u062a|\u062e\u062f\u062a|\u0623\u062e\u0630\u062a|\u062d\u0627\u0633\u0633|\u062d\u0627\u0633\u0647)'
            r'|\b(?:akhat|akht|5adt)\b',
            text,
        ):
            return True

        # Conservative default for bare/unclear "overdose" mention without educational indicators
        return True

    return False



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


def _empty_conversation_result(language, language_style):
    return {
        'language': language,
        'language_style': language_style,
        'extracted': [],
        'resolved': [],
        'verified_generics': [],
        'retrievals': [],
        'response': None,
    }


def _conversation_fallback(language_style):
    if language_style == 'arabic':
        return 'من فضلك اذكر أسماء الأدوية التي تريد التحقق منها.'
    if language_style == 'arabizi':
        return 'Momken tekteb asamy el adweya elly 3awez tet2aked menha?'
    if language_style == 'mixed':
        return 'من فضلك اذكر أسماء الأدوية اللي عايز تتأكد منها.'
    return 'Please tell me the names of the medications you want to check.'


@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        # --- Rate Limiting ---
        rate_key = request.headers.get('X-Forwarded-For') or request.remote_addr or 'unknown'
        if not RATE_LIMITER.is_allowed(rate_key):
            resp = jsonify({'error': 'Rate limit exceeded. Please wait before sending more requests.'})
            resp.headers['Retry-After'] = '60'
            return resp, 429

        # --- Payload & Message Size Limits ---
        if request.content_length and request.content_length > MAX_PAYLOAD_BYTES:
            return jsonify({'error': f'Request payload exceeds maximum allowed size of {MAX_PAYLOAD_BYTES} bytes.'}), 413

        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON format. Expected JSON object.'}), 400

        message = data.get('message', '')
        if not message or not str(message).strip():
            return jsonify({'error': 'Empty message'}), 400

        message = str(message).strip()
        if len(message) > MAX_MESSAGE_LENGTH:
            return jsonify({
                'error': f'Message length ({len(message)} characters) exceeds the maximum allowed limit of {MAX_MESSAGE_LENGTH} characters.'
            }), 413

        # --- Client & Session Identification ---
        client_id, is_new_client = _get_or_create_client_id()
        session_id = data.get('session_id') if isinstance(data, dict) else None
        try:
            conversation_context, session_key = _session_context(session_id, client_id)
        except SessionSecurityError as e:
            return jsonify({'error': e.message}), e.status_code

        mod = rag

        # --- Safety Intent Check ---
        if _has_safety_intent(message):
            language_style = mod.detect_language_style(message)
            response_text = _localized_guardrail_response(language_style, 'safety')
            resp = {
                'message': message,
                'session_id': session_key,
                'language': mod.detect_language(message),
                'language_style': language_style,
                'extracted': [],
                'resolved': [],
                'verified_generics': [],
                'retrievals': [],
                'response': response_text,
                'needs_clarification': False,
                'groq_used': False,
            }
            response_obj = jsonify(resp)
            if is_new_client:
                _set_client_cookie(response_obj, client_id)
            return response_obj

        # --- Out-of-scope Intent Check ---
        # Runs after safety (always highest priority) but before the medication
        # pipeline.  Fully deterministic — requires no Groq API call.
        if _is_out_of_scope(message, mod):
            language_style = mod.detect_language_style(message)
            resp = {
                'message': message,
                'session_id': session_key,
                'language': mod.detect_language(message),
                'language_style': language_style,
                'extracted': [],
                'resolved': [],
                'verified_generics': [],
                'retrievals': [],
                'response': _localized_guardrail_response(language_style, 'out_of_scope'),
                'needs_clarification': False,
                'groq_used': False,
                'out_of_scope': True,
            }
            response_obj = jsonify(resp)
            if is_new_client:
                _set_client_cookie(response_obj, client_id)
            return response_obj

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
            result = _empty_conversation_result(
                mod.detect_language(message),
                mod.detect_language_style(message),
            )
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
                result['language_style'] = mod.detect_language_style(message)

                if session_key:
                    active = conversation_context.get('active_medications', [])
                    for medication in result.get('verified_generics', []):
                        if medication not in active:
                            active.append(medication)
            except Exception:
                logger.error("Failed to process medication pipeline; falling back to basic processing", exc_info=True)
                result = rag.process_message(message, conversation_context=conversation_context)
                result['language'] = rag.detect_language(message)
                result['language_style'] = rag.detect_language_style(message)

            needs_clarification = (
                any(not r.get('verified') for r in result.get('resolved', []))
                and _is_medication_query(message, result)
            )

        groq_response = None
        if not needs_clarification:
            session_history = conversation_context.get('history', []) if session_key else []
            groq_response = generate_reply(message, result, session_history)
            if groq_response and not matches_language_style(
                groq_response,
                result.get('language_style', 'english'),
            ):
                print('Generated response rejected: wrong language style')
                groq_response = None
        response = groq_response or result.get('response')
        if response is None and not medication_message:
            response = _conversation_fallback(result.get('language_style', 'english'))

        if session_key:
            history = conversation_context.setdefault('history', [])
            history.append({'role': 'user', 'content': message})
            history.append({'role': 'assistant', 'content': response})
            del history[:-MAX_HISTORY_MESSAGES]

        resp = {
            'message': message,
            'session_id': session_key,
            'language': result.get('language'),
            'language_style': result.get('language_style', result.get('language')),
            'extracted': result.get('extracted'),
            'resolved': result.get('resolved'),
            'verified_generics': result.get('verified_generics'),
            'retrievals': result.get('retrievals'),
            'response': response,
            'needs_clarification': needs_clarification,
            'groq_used': bool(groq_response),
        }
        response_obj = jsonify(resp)
        if is_new_client:
            _set_client_cookie(response_obj, client_id)
        return response_obj

    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/session', methods=['DELETE'])
def delete_session():
    """Delete a session's state. Called by the frontend on Clear."""
    try:
        # --- Rate Limiting ---
        rate_key = request.headers.get('X-Forwarded-For') or request.remote_addr or 'unknown'
        if not RATE_LIMITER.is_allowed(rate_key):
            resp = jsonify({'error': 'Rate limit exceeded. Please wait before sending more requests.'})
            resp.headers['Retry-After'] = '60'
            return resp, 429

        if request.content_length and request.content_length > MAX_PAYLOAD_BYTES:
            return jsonify({'error': 'Request payload exceeds maximum allowed size.'}), 413

        client_id, is_new_client = _get_or_create_client_id()
        data = request.get_json(force=True) if request.data else {}
        session_id = data.get('session_id') if isinstance(data, dict) else None
        if session_id and session_id in SESSIONS:
            owner = SESSIONS[session_id].get('owner_id')
            if owner and client_id and owner != client_id:
                return jsonify({'error': 'Unauthorized: Cannot delete session belonging to another client'}), 403
            del SESSIONS[session_id]

        response_obj = jsonify({'status': 'ok'})
        if is_new_client:
            _set_client_cookie(response_obj, client_id)
        return response_obj
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

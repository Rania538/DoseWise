import json
import sys
import pytest
from src import groq_chat
from src import rag_pipeline
from src import server


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_groq(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'fake_key')

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)

    class FakeCompletion:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeClient:
        last_messages = None

        def __init__(self, api_key=None):
            self.api_key = api_key

        class chat:
            class completions:
                @classmethod
                def create(cls, model, messages, temperature=0.2, max_tokens=500):
                    FakeClient.last_messages = messages
                    return FakeCompletion("Here is the requested information.")

    fake_mod = type(sys)('groq')
    fake_mod.Groq = FakeClient
    monkeypatch.setitem(sys.modules, 'groq', fake_mod)
    return FakeClient


@pytest.fixture
def client():
    server.SESSIONS.clear()
    server._reset_rate_limits()
    with server.app.test_client() as c:
        yield c
    server.SESSIONS.clear()
    server._reset_rate_limits()


# ---------------------------------------------------------------------------
# 1. Structured Grounding Contract Tests
# ---------------------------------------------------------------------------

def test_evidence_context_structure_with_interaction():
    """Verify that _evidence_context provides strict structured fields when interaction is present."""
    result = {
        'extracted': ['aspirin', 'warfarin'],
        'resolved': [
            {'input': 'aspirin', 'verified': True, 'generic_name': 'aspirin'},
            {'input': 'warfarin', 'verified': True, 'generic_name': 'warfarin'},
        ],
        'verified_generics': ['aspirin', 'warfarin'],
        'retrievals': [
            {
                'drug_a': 'aspirin',
                'drug_b': 'warfarin',
                'interaction_level': 'major',
                'ddinter_rows': 2,
                'chunk_ids': ['DDInter_001', 'DDInter_002'],
            }
        ],
    }
    raw_json = groq_chat._evidence_context(result)
    parsed = json.loads(raw_json)

    assert parsed['source'] == 'DDInter'
    assert parsed['interaction_found'] is True
    assert parsed['interaction_severity'] == ['Major']
    assert 'aspirin' in parsed['verified_active_ingredients']
    assert 'warfarin' in parsed['verified_active_ingredients']
    assert len(parsed['ddinter_records']) == 1
    assert parsed['ddinter_records'][0]['interaction_level'] == 'major'
    assert parsed['ddinter_records'][0]['chunk_ids'] == ['DDInter_001', 'DDInter_002']


def test_evidence_context_structure_without_interaction():
    """Verify that _evidence_context correctly marks interaction_found=False and empty severity when none found."""
    result = {
        'extracted': ['drugx', 'drugy'],
        'resolved': [
            {'input': 'drugx', 'verified': True, 'generic_name': 'drugx'},
            {'input': 'drugy', 'verified': True, 'generic_name': 'drugy'},
        ],
        'verified_generics': ['drugx', 'drugy'],
        'retrievals': [
            {
                'drug_a': 'drugx',
                'drug_b': 'drugy',
                'interaction_level': None,
                'ddinter_rows': 0,
                'chunk_ids': [],
            }
        ],
    }
    raw_json = groq_chat._evidence_context(result)
    parsed = json.loads(raw_json)

    assert parsed['source'] == 'DDInter'
    assert parsed['interaction_found'] is False
    assert parsed['interaction_severity'] == []
    assert parsed['ddinter_records'][0]['interaction_level'] is None


def test_groq_receives_strict_grounding_contract(mock_groq):
    """Verify that Groq receives the structured grounding contract in its system prompt."""
    result = {
        'extracted': ['aspirin', 'warfarin'],
        'resolved': [
            {'input': 'aspirin', 'verified': True, 'generic_name': 'aspirin'},
            {'input': 'warfarin', 'verified': True, 'generic_name': 'warfarin'},
        ],
        'verified_generics': ['aspirin', 'warfarin'],
        'retrievals': [
            {
                'drug_a': 'aspirin',
                'drug_b': 'warfarin',
                'interaction_level': 'major',
                'ddinter_rows': 1,
                'chunk_ids': ['DDInter_001'],
            }
        ],
        'language_style': 'english',
    }

    reply = groq_chat.generate_reply("Can I take aspirin with warfarin?", result)
    assert reply is not None

    messages = mock_groq.last_messages
    assert messages is not None
    system_prompt = messages[0]['content']

    # Must contain the explicit Grounding Contract rules
    assert 'GROUNDING CONTRACT:' in system_prompt
    assert 'DDInter is the SOLE source of medical truth' in system_prompt
    assert 'Never invent an interaction, severity, contraindication, symptom, dosage advice, or medical fact' in system_prompt
    assert 'Severity must ONLY be reported if present in interaction_severity' in system_prompt
    assert 'CRITICAL SAFETY BOUNDARY:' in system_prompt


# ---------------------------------------------------------------------------
# 2. "No Interaction Found" Safety Boundary Tests
# ---------------------------------------------------------------------------

def test_no_interaction_system_prompt_forbids_claiming_universal_safety(mock_groq):
    """The system prompt must explicitly instruct Groq never to claim universal safety when no interaction is found."""
    result = {
        'extracted': ['drugx', 'drugy'],
        'resolved': [
            {'input': 'drugx', 'verified': True, 'generic_name': 'drugx'},
            {'input': 'drugy', 'verified': True, 'generic_name': 'drugy'},
        ],
        'verified_generics': ['drugx', 'drugy'],
        'retrievals': [
            {
                'drug_a': 'drugx',
                'drug_b': 'drugy',
                'interaction_level': None,
                'ddinter_rows': 0,
                'chunk_ids': [],
            }
        ],
        'language_style': 'english',
    }

    groq_chat.generate_reply("Can I take drugx with drugy?", result)
    system_prompt = mock_groq.last_messages[0]['content']

    assert 'no interaction was identified in the available DDInter knowledge base' in system_prompt
    assert 'NEVER claim or imply that the medications are "completely safe"' in system_prompt


def test_deterministic_response_for_no_interaction_does_not_claim_universal_safety():
    """Verify that the deterministic RAG response states no interaction found without claiming 100% safety."""
    retrievals = [
        {
            'drug_a': 'paracetamol',
            'drug_b': 'vitamin c',
            'interaction_level': None,
            'ddinter_rows': 0,
            'chunk_ids': [],
        }
    ]
    resolved = [
        {'input': 'paracetamol', 'verified': True, 'generic_name': 'paracetamol'},
        {'input': 'vitamin c', 'verified': True, 'generic_name': 'ascorbic acid'},
    ]

    resp_en = rag_pipeline.generate_response("Can I take paracetamol with vitamin c?", 'english', resolved, retrievals)
    assert "No interaction was identified in the available DDInter knowledge base" in resp_en
    assert "completely safe" not in resp_en.lower()
    assert "100% safe" not in resp_en.lower()
    assert "no risk" not in resp_en.lower()

    # Safety note must be present
    assert "Do not stop or change a prescribed medication without consulting your doctor or pharmacist" in resp_en


# ---------------------------------------------------------------------------
# 3. General Health Question Boundary Tests
# ---------------------------------------------------------------------------

def test_general_health_question_uses_conversational_mode_without_ddinter_evidence(mock_groq):
    """General health questions (e.g. 'What causes headaches?') must use conversational mode without DDInter evidence."""
    empty_result = {
        'extracted': [],
        'resolved': [],
        'verified_generics': [],
        'retrievals': [],
        'language_style': 'english',
    }

    groq_chat.generate_reply("What causes headaches?", empty_result)
    messages = mock_groq.last_messages
    assert messages is not None
    system_prompt = messages[0]['content']

    # Must contain knowledge boundary rules
    assert 'KNOWLEDGE BOUNDARIES:' in system_prompt
    assert 'NEVER claim or imply that general health answers come from DDInter records' in system_prompt
    assert 'NEVER make clinical diagnoses, prescribe medications, or provide dosage advice' in system_prompt
    assert 'advise the user to consult a doctor or healthcare professional' in system_prompt

    # Must NOT contain medication specialist evidence injection
    assert 'Here is the current structured evidence:' not in system_prompt


def test_general_health_api_flow_without_groq(client, monkeypatch):
    """When Groq is unavailable, general health query without medications falls back safely to medication prompt."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: None)
    res = client.post('/api/chat', json={'message': 'What is dehydration?'})
    assert res.status_code == 200
    data = res.get_json()

    # Must NOT claim DDInter evidence
    assert data['retrievals'] == []
    assert data['verified_generics'] == []
    assert data['groq_used'] is False
    # Safe fallback message prompts for medication names
    assert 'Please tell me the names of the medications you want to check.' in data['response']


def test_general_health_api_flow_with_groq_boundary(client, monkeypatch):
    """When Groq is used, general health responses must not claim DDInter evidence and must recommend healthcare consultation."""
    monkeypatch.setattr(
        server,
        'generate_reply',
        lambda msg, res, hist: 'Dehydration means losing fluids. Please consult a doctor for diagnosis.'
    )
    res = client.post('/api/chat', json={'message': 'What is dehydration?'})
    assert res.status_code == 200
    data = res.get_json()

    assert data['retrievals'] == []
    assert data['verified_generics'] == []
    assert data['groq_used'] is True
    assert 'DDInter' not in data['response']
    assert 'consult a doctor' in data['response']


# ---------------------------------------------------------------------------
# 4. Multi-Ingredient and Safety Regression Tests
# ---------------------------------------------------------------------------

def test_combination_medication_active_ingredients_preserved(monkeypatch):
    """Verify that multi-ingredient medication resolution preserves all active ingredients."""
    def fake_resolve(extracted):
        return [
            {
                'input': 'combo_antibiotic',
                'generic_name': 'amoxicillin / clavulanate potassium',
                'active_ingredients': ['amoxicillin', 'clavulanate potassium'],
                'verified': True,
                'source': 'local_alias',
                'match_type': 'alias',
            }
        ]

    monkeypatch.setattr(rag_pipeline, 'resolve_medications', fake_resolve)
    monkeypatch.setattr(rag_pipeline, 'extract_medications', lambda msg: ['combo_antibiotic'])

    out = rag_pipeline.process_message("Can I take combo_antibiotic?")
    assert 'amoxicillin' in out['verified_generics']
    assert 'clavulanate potassium' in out['verified_generics']
    assert len(out['verified_generics']) == 2


def test_safety_guardrails_educational_vs_emergency(client):
    """Verify Phase 2 safety behavior remains intact: educational is non-emergency, ingestion error is emergency."""
    # Educational query
    res_edu = client.post('/api/chat', json={'message': 'What is an overdose?'})
    assert res_edu.status_code == 200
    data_edu = res_edu.get_json()
    # Does not trigger emergency poison control response
    assert 'potential medical emergency' not in data_edu['response']

    # Active ingestion / error query
    res_emerg = client.post('/api/chat', json={'message': 'I took 5 pills by mistake'})
    assert res_emerg.status_code == 200
    data_emerg = res_emerg.get_json()
    # Triggers emergency safety guardrail
    assert 'potential medical emergency or overdose' in data_emerg['response']
    assert data_emerg['groq_used'] is False


# ---------------------------------------------------------------------------
# 5. Mocked Bad-Groq-Response Rejection Tests
#    These verify that validate_grounded_reply (wired into generate_reply)
#    rejects unsupported claims before they reach the user.
# ---------------------------------------------------------------------------

def _make_groq_module(monkeypatch, reply_content: str):
    """Patch sys.modules['groq'] so Groq always returns reply_content."""
    import sys

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)

    class FakeCompletion:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class chat:
            class completions:
                @classmethod
                def create(cls, model, messages, temperature=0.2, max_tokens=500):
                    return FakeCompletion(reply_content)

    monkeypatch.setenv('GROQ_API_KEY', 'fake_key')
    fake_mod = type(sys)('groq')
    fake_mod.Groq = FakeClient
    monkeypatch.setitem(sys.modules, 'groq', fake_mod)


# Shared result fixture: two medications, NO interaction in DDInter
_NO_INTERACTION_RESULT = {
    'extracted': ['drugx', 'drugy'],
    'resolved': [
        {'input': 'drugx', 'verified': True, 'generic_name': 'drugx'},
        {'input': 'drugy', 'verified': True, 'generic_name': 'drugy'},
    ],
    'verified_generics': ['drugx', 'drugy'],
    'retrievals': [
        {
            'drug_a': 'drugx',
            'drug_b': 'drugy',
            'interaction_level': None,
            'ddinter_rows': 0,
            'chunk_ids': [],
        }
    ],
    'language_style': 'english',
}

# Shared result fixture: two medications, MAJOR interaction in DDInter
_MAJOR_INTERACTION_RESULT = {
    'extracted': ['aspirin', 'warfarin'],
    'resolved': [
        {'input': 'aspirin', 'verified': True, 'generic_name': 'aspirin'},
        {'input': 'warfarin', 'verified': True, 'generic_name': 'warfarin'},
    ],
    'verified_generics': ['aspirin', 'warfarin'],
    'retrievals': [
        {
            'drug_a': 'aspirin',
            'drug_b': 'warfarin',
            'interaction_level': 'major',
            'ddinter_rows': 2,
            'chunk_ids': ['DDInter_001'],
        }
    ],
    'language_style': 'english',
}


def test_mocked_groq_invented_interaction_rejected(monkeypatch):
    """When DDInter found NO interaction, a Groq reply claiming 'interaction detected' must be rejected (returns None)."""
    _make_groq_module(
        monkeypatch,
        'DrugX and DrugY have a known interaction detected in our database.',
    )
    result = groq_chat.generate_reply('Can I take drugx with drugy?', _NO_INTERACTION_RESULT)
    assert result is None, (
        'generate_reply should return None when Groq invents an interaction not found in DDInter'
    )


def test_mocked_groq_invented_severity_rejected(monkeypatch):
    """When DDInter evidence only has a major interaction, a Groq reply claiming 'moderate severity' must be rejected."""
    _make_groq_module(
        monkeypatch,
        'The severity: moderate interaction between aspirin and warfarin may cause mild bleeding.',
    )
    result = groq_chat.generate_reply('Can I take aspirin with warfarin?', _MAJOR_INTERACTION_RESULT)
    assert result is None, (
        'generate_reply should return None when Groq asserts a severity (moderate) not present in DDInter evidence (major only)'
    )


def test_mocked_groq_unsupported_contraindication_rejected(monkeypatch):
    """A Groq reply asserting a formal contraindication must be rejected."""
    _make_groq_module(
        monkeypatch,
        'Aspirin is strictly contraindicated with warfarin and must never be co-administered.',
    )
    result = groq_chat.generate_reply('Can I take aspirin with warfarin?', _MAJOR_INTERACTION_RESULT)
    assert result is None, (
        'generate_reply should return None when Groq asserts a contraindication not supported by DDInter'
    )


def test_mocked_groq_unsupported_dosage_advice_rejected(monkeypatch):
    """A Groq reply giving prescriptive dosage instructions (take 500mg) must be rejected."""
    _make_groq_module(
        monkeypatch,
        'You should take 500mg of aspirin twice daily and adjust dose to 75mg if bleeding occurs.',
    )
    result = groq_chat.generate_reply('Can I take aspirin with warfarin?', _MAJOR_INTERACTION_RESULT)
    assert result is None, (
        'generate_reply should return None when Groq provides prescriptive dosage advice'
    )


def test_mocked_groq_universal_safety_claim_rejected(monkeypatch):
    """When DDInter found no interaction, a Groq reply claiming 'completely safe' must be rejected."""
    _make_groq_module(
        monkeypatch,
        'DrugX and DrugY are completely safe to take together without any risk.',
    )
    result = groq_chat.generate_reply('Can I take drugx with drugy?', _NO_INTERACTION_RESULT)
    assert result is None, (
        'generate_reply should return None when Groq claims medications are completely safe'
    )


def test_mocked_groq_fabricated_ddinter_citation_in_general_mode_rejected(monkeypatch):
    """A Groq reply in general health mode that fabricates a DDInter citation must be rejected."""
    _make_groq_module(
        monkeypatch,
        'According to DDInter record DDI-9999, headaches are caused by dehydration.',
    )
    empty_result = {
        'extracted': [],
        'resolved': [],
        'verified_generics': [],
        'retrievals': [],
        'language_style': 'english',
    }
    result = groq_chat.generate_reply('What causes headaches?', empty_result)
    assert result is None, (
        'generate_reply should return None when Groq fabricates a DDInter citation in general conversational mode'
    )


def test_mocked_groq_valid_grounded_response_passes(monkeypatch):
    """A Groq reply that correctly describes a major interaction without forbidden claims must pass through."""
    _make_groq_module(
        monkeypatch,
        (
            '💊 Medications Detected: aspirin, warfarin\n'
            '🔍 Interaction Result: A major interaction was identified in DDInter.\n'
            '⚠️ Severity: Major\n'
            '📌 Explanation: Concurrent use may increase bleeding risk.\n'
            '⚠️ Safety Note: Please consult your doctor or pharmacist before making any changes.'
        ),
    )
    result = groq_chat.generate_reply('Can I take aspirin with warfarin?', _MAJOR_INTERACTION_RESULT)
    assert result is not None, (
        'generate_reply should return a valid grounded reply when Groq response contains only supported claims'
    )
    assert 'major' in result.lower()


def test_mocked_groq_no_interaction_grounded_response_passes(monkeypatch):
    """A Groq reply correctly stating no DDInter interaction was found (without claiming safety) must pass through."""
    _make_groq_module(
        monkeypatch,
        (
            '💊 Medications Detected: drugx, drugy\n'
            '🔍 Interaction Result: No interaction was identified in the available DDInter knowledge base.\n'
            '⚠️ Safety Note: Absence of a recorded interaction does not confirm these medications are safe to combine. '
            'Please consult your doctor or pharmacist.'
        ),
    )
    result = groq_chat.generate_reply('Can I take drugx with drugy?', _NO_INTERACTION_RESULT)
    assert result is not None, (
        'generate_reply should return a valid grounded reply when Groq correctly states no interaction found'
    )
    assert 'no interaction' in result.lower()


# ---------------------------------------------------------------------------
# 3. Conversational Quality & Repetition Guard Tests
# ---------------------------------------------------------------------------

def test_pathological_repetition_detection_flagging():
    """Verify has_pathological_repetition catches 3+ consecutive repeated phrases without false positives."""
    # Repetitive loops should be flagged:
    assert groq_chat.has_pathological_repetition("Ana DoseWise, mas'ad fi el swala el 3awya el 3awya el 3awya...")
    assert groq_chat.has_pathological_repetition("test phrase, test phrase, test phrase")
    assert groq_chat.has_pathological_repetition("word word word")

    # Natural language and legitimate medical phrasing should NOT be flagged:
    assert not groq_chat.has_pathological_repetition("very very good")
    assert not groq_chat.has_pathological_repetition("Hello! How can I help you with your medications today?")
    assert not groq_chat.has_pathological_repetition("Ana kwayes elhamdolellah, shokran! A2dar asa3dak ezay fe el adweya ennaharda?")
    assert not groq_chat.has_pathological_repetition("أنا بخير والحمد لله، شكراً لسؤالك! كيف أقدر أساعدك اليوم في أدويتك؟")
    assert not groq_chat.has_pathological_repetition(
        "💊 Medications Detected: Panadol (acetaminophen), Amoxicillin\n"
        "🔍 Interaction Result: No interaction was identified in the available DDInter knowledge base.\n"
        "⚠️ Severity: None\n"
        "📌 Explanation: Acetaminophen and amoxicillin can generally be taken together as prescribed.\n"
        "📚 Evidence: DDInter records.\n"
        "⚠️ Safety Note: Please consult your doctor or pharmacist."
    )


def test_mocked_groq_pathological_repetition_rejected(monkeypatch):
    """When Groq returns a response containing degenerate repetition, generate_reply must reject it."""
    repetitive_reply = "Ana DoseWise, mas'ad fi el swala el 3awya el 3awya el 3awya..."
    _make_groq_module(monkeypatch, repetitive_reply)

    arabizi_result = {
        'extracted': [],
        'resolved': [],
        'verified_generics': [],
        'retrievals': [],
        'language_style': 'arabizi',
    }
    result = groq_chat.generate_reply('enta kwis?', arabizi_result)
    assert result is None, "generate_reply must return None when Groq reply contains pathological repetition"


def test_conversational_enta_kwis_clean_reply(monkeypatch):
    """'enta kwis?' with a clean Arabizi response must pass validation and style checks."""
    valid_arabizi = "Ana kwayes elhamdolellah, shokran! A2dar asa3dak ezay fe el adweya ennaharda?"
    _make_groq_module(monkeypatch, valid_arabizi)

    arabizi_result = {
        'extracted': [],
        'resolved': [],
        'verified_generics': [],
        'retrievals': [],
        'language_style': 'arabizi',
    }
    result = groq_chat.generate_reply('enta kwis?', arabizi_result)
    assert result is not None
    assert result == valid_arabizi
    assert not groq_chat.has_pathological_repetition(result)


def test_conversational_hello_clean_reply(monkeypatch):
    """'hello' with a clean English response must pass validation and style checks."""
    valid_english = "Hello! How can I help you with your medications today?"
    _make_groq_module(monkeypatch, valid_english)

    english_result = {
        'extracted': [],
        'resolved': [],
        'verified_generics': [],
        'retrievals': [],
        'language_style': 'english',
    }
    result = groq_chat.generate_reply('hello', english_result)
    assert result is not None
    assert result == valid_english
    assert not groq_chat.has_pathological_repetition(result)


def test_conversational_arabic_message_clean_reply(monkeypatch):
    """Arabic conversational greeting with clean Arabic response must pass validation and style checks."""
    valid_arabic = "أنا بخير والحمد لله، شكراً لسؤالك! كيف أقدر أساعدك اليوم في أدويتك؟"
    _make_groq_module(monkeypatch, valid_arabic)

    arabic_result = {
        'extracted': [],
        'resolved': [],
        'verified_generics': [],
        'retrievals': [],
        'language_style': 'arabic',
    }
    result = groq_chat.generate_reply('أهلاً، عامل إيه؟', arabic_result)
    assert result is not None
    assert result == valid_arabic
    assert not groq_chat.has_pathological_repetition(result)


def test_conversational_franco_arabic_clean_reply(monkeypatch):
    """Franco-Arabic / Arabizi message with clean Arabizi response must pass validation and style checks."""
    valid_franco = "Ana kwayes elhamdolellah, shokran! A2dar asa3dak ezay fe el adweya?"
    _make_groq_module(monkeypatch, valid_franco)

    arabizi_result = {
        'extracted': [],
        'resolved': [],
        'verified_generics': [],
        'retrievals': [],
        'language_style': 'arabizi',
    }
    result = groq_chat.generate_reply('ezayak 3amel eh?', arabizi_result)
    assert result is not None
    assert result == valid_franco
    assert not groq_chat.has_pathological_repetition(result)


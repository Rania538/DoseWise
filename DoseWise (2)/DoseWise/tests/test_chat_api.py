from src import server
from src.groq_chat import _clean_reply


def post_chat(monkeypatch, message):
    replies = []

    def fake_groq_reply(user_message, result, history):
        replies.append(user_message)
        return f'Groq reply: {user_message}'

    monkeypatch.setattr(server, 'generate_reply', fake_groq_reply)
    server.SESSIONS.clear()
    with server.app.test_client() as client:
        response = client.post('/api/chat', json={'message': message})
    return response.get_json(), replies


def test_normal_introduction_uses_groq_without_clarification(monkeypatch):
    result, replies = post_chat(monkeypatch, 'hi my name is ranya')

    assert result['needs_clarification'] is False
    assert result['groq_used'] is True
    assert replies == ['hi my name is ranya']


def test_hello_uses_groq_without_clarification(monkeypatch):
    result, replies = post_chat(monkeypatch, 'hello')

    assert result['needs_clarification'] is False
    assert result['groq_used'] is True
    assert replies == ['hello']


def test_normal_conversation_messages_bypass_medication_pipeline(monkeypatch):
    replies = []

    def fake_groq_reply(user_message, result, history):
        replies.append(user_message)
        return 'Natural reply'

    monkeypatch.setattr(server, 'generate_reply', fake_groq_reply)
    server.SESSIONS.clear()
    session_id = 'mixed-conversation-test'

    with server.app.test_client() as client:
        medication = client.post('/api/chat', json={
            'message': 'Can I take Panadol with amoxicillin?',
            'session_id': session_id,
        }).get_json()
        normal = client.post('/api/chat', json={
            'message': 'Thanks!',
            'session_id': session_id,
        }).get_json()

    assert len(medication['retrievals']) == 1
    assert normal['extracted'] == []
    assert normal['resolved'] == []
    assert normal['verified_generics'] == []
    assert normal['retrievals'] == []
    assert normal['needs_clarification'] is False
    assert normal['groq_used'] is True
    assert replies == ['Can I take Panadol with amoxicillin?', 'Thanks!']


def test_arabic_and_arabizi_medication_questions_use_rag(monkeypatch):
    monkeypatch.setattr(server, 'generate_reply', lambda user_message, result, history: 'ok')
    server.SESSIONS.clear()

    with server.app.test_client() as client:
        arabic = client.post('/api/chat', json={
            'message': 'انا باخد بانادول مع amoxicillin',
        }).get_json()
        arabizi = client.post('/api/chat', json={
            'message': 'ana 5dt Pandol w amoxicilin, ynf3 akhodhom m3 ba3d?',
        }).get_json()

    for result in (arabic, arabizi):
        assert result['needs_clarification'] is False
        assert set(result['verified_generics']) >= {'acetaminophen', 'amoxicillin'}
        assert result['retrievals']


def test_medication_question_keeps_rag_flow(monkeypatch):
    result, replies = post_chat(monkeypatch, 'Can I take Panadol with amoxicillin?')

    assert result['needs_clarification'] is False
    assert result['groq_used'] is True
    assert set(result['verified_generics']) >= {'acetaminophen', 'amoxicillin'}
    assert len(result['retrievals']) == 1
    assert replies == ['Can I take Panadol with amoxicillin?']





def test_unknown_medication_keeps_clarification_flow(monkeypatch):
    result, replies = post_chat(monkeypatch, 'XyzUnknown and amoxicillin')

    assert result['needs_clarification'] is True
    assert 'Did you mean' in result['response']
    assert replies == []


def test_clean_reply_removes_malformed_reasoning_block():
    assert _clean_reply('think>reasoning...</think>Useful answer.') == 'Useful answer.'
    assert _clean_reply('<think>reasoning...</think>') is None


def test_groq_system_prompt_is_conversational_for_greeting(monkeypatch):
    """When generate_reply is called with an empty result (no meds), it must use
    the conversational system prompt and must NOT inject DDInter evidence text."""
    captured_messages = []

    class FakeCompletion:
        class choices:
            class _choice:
                class message:
                    content = 'Hello!'
            choices = [_choice()]

    class FakeClient:
        def __init__(self, api_key):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature, max_tokens):
                    captured_messages.extend(messages)
                    return FakeCompletion

    import src.groq_chat as gc
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    monkeypatch.setattr(gc, 'Groq', FakeClient, raising=False)

    # Patch the import inside generate_reply too
    import importlib, sys
    orig_groq = sys.modules.get('groq')
    fake_groq_mod = type(sys)('groq')
    fake_groq_mod.Groq = FakeClient
    sys.modules['groq'] = fake_groq_mod
    try:
        empty_result = {'verified_generics': [], 'retrievals': [], 'language': 'en'}
        gc.generate_reply('hi my name is ranya', empty_result, history=[])
    finally:
        if orig_groq is None:
            sys.modules.pop('groq', None)
        else:
            sys.modules['groq'] = orig_groq

    assert captured_messages, 'Groq was not called'
    system_msg = captured_messages[0]
    assert system_msg['role'] == 'system'
    # Must NOT contain medication-specialist framing
    assert 'DDInter evidence' not in system_msg['content']
    assert 'pipeline has already identified' not in system_msg['content']
    assert 'friendly' in system_msg['content'] or 'health assistant' in system_msg['content']


def test_context_reset_on_fresh_query(monkeypatch):
    import src.server as server
    server.SESSIONS.clear()
    with server.app.test_client() as client:
        # 1. "Can I take Panadol with amoxicillin?"
        client.post('/api/chat', json={'session_id': 'ctx_fresh', 'message': 'Can I take Panadol with amoxicillin?'})
        
        # 2. "Can I take ibuprofen with metformin?" (Fresh query)
        res2 = client.post('/api/chat', json={'session_id': 'ctx_fresh', 'message': 'Can I take ibuprofen with metformin?'})
        data = res2.get_json()
        
        # The second message should NOT include acetaminophen/amoxicillin
        generics = [g.lower() for g in data.get('verified_generics', [])]
        assert 'acetaminophen' not in generics
        assert 'amoxicillin' not in generics
        assert 'ibuprofen' in generics
        assert 'metformin' in generics


def test_context_followup_inherits(monkeypatch):
    import src.server as server
    server.SESSIONS.clear()
    with server.app.test_client() as client:
        # 1. "Can I take Panadol with amoxicillin?"
        client.post('/api/chat', json={'session_id': 'ctx_follow', 'message': 'Can I take Panadol with amoxicillin?'})
        
        # 2. "What about simvastatin?" (Follow-up)
        res2 = client.post('/api/chat', json={'session_id': 'ctx_follow', 'message': 'What about simvastatin?'})
        data = res2.get_json()
        
        # UI shows ONLY the new medication for the second message
        generics = [g.lower() for g in data.get('verified_generics', [])]
        assert 'simvastatin' in generics
        assert 'acetaminophen' not in generics
        assert 'amoxicillin' not in generics
        
        # Retrievals must query the pairs (acetaminophen+simvastatin, amoxicillin+simvastatin)
        pairs = {frozenset((r['drug_a'].lower(), r['drug_b'].lower())) for r in data['retrievals']}
        assert frozenset(('acetaminophen', 'simvastatin')) in pairs
        assert frozenset(('amoxicillin', 'simvastatin')) in pairs


def test_arabic_followup_tab_w_simvastatin(monkeypatch):
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')
    server.SESSIONS.clear()
    with server.app.test_client() as client:
        # Initial pair
        client.post('/api/chat', json={'session_id': 'sess_ar', 'message': 'Panadol and amoxicillin'})
        
        # Arabic follow-up
        res = client.post('/api/chat', json={'session_id': 'sess_ar', 'message': 'طب والسيمفاستاتين؟'})
        data = res.get_json()
        
        generics = [g.lower() for g in data.get('verified_generics', [])]
        assert 'simvastatin' in generics
        assert 'acetaminophen' not in generics
        
        pairs = {frozenset((r['drug_a'].lower(), r['drug_b'].lower())) for r in data['retrievals']}
        assert frozenset(('acetaminophen', 'simvastatin')) in pairs
        assert frozenset(('amoxicillin', 'simvastatin')) in pairs


def test_clear_session_isolation(monkeypatch):
    import src.server as server
    server.SESSIONS.clear()
    with server.app.test_client() as client:
        # Session A gets meds
        client.post('/api/chat', json={'session_id': 'sessA', 'message': 'Panadol and amoxicillin'})
        
        # New session B queries simvastatin without follow-up prefix
        # Should NOT inherit sessA meds
        resB = client.post('/api/chat', json={'session_id': 'sessB', 'message': 'simvastatin'})
        dataB = resB.get_json()
        genericsB = [g.lower() for g in dataB.get('verified_generics', [])]
        assert 'simvastatin' in genericsB
        assert 'acetaminophen' not in genericsB
        
        # Delete Session A via DELETE API
        client.delete('/api/session', json={'session_id': 'sessA'})
        assert 'sessA' not in server.SESSIONS


def test_tab_arabic_not_medication(monkeypatch):
    monkeypatch.setattr(server, 'generate_reply', lambda user_message, result, history: 'normal chat')
    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'طب'})
        data = res.get_json()
        assert not data.get('verified_generics')
        assert data['response'] == 'normal chat'


def test_safety_intent_bypasses_clarification():
    queries = [
        "I took 5 Panadol tablets at once. Should I take more?",
        "I accidentally took an extra dose.",
        "Can I double my dose?",
        "I took too many tablets."
    ]
    
    with server.app.test_client() as client:
        for q in queries:
            res = client.post('/api/chat', json={'message': q})
            data = res.get_json()
            assert data['needs_clarification'] is False
            assert data['groq_used'] is False
            assert "emergency or overdose" in data['response']
            assert "Do NOT take any more medication" in data['response']


# ---------------------------------------------------------------------------
# Out-of-scope guardrail tests
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_QUERIES = [
    "Tell me about Ahmed Zewail",
    "Who is Ahmed Zewail?",
    "What's the capital of Egypt?",
    "Write me a Python program",
    "Tell me a joke",
]

_IN_SCOPE_QUERIES = [
    "What is Panadol?",
    "Can I take Panadol with amoxicillin?",
    "I accidentally took too much Panadol.",
]

_CONVERSATIONAL_QUERIES = [
    "Hi, my name is Ranya",
    "Thanks",
    "Hello, how are you?",
]


def _patch_scope_classifier_returns(monkeypatch, return_value: bool):
    """Replace the deterministic scope classifier with a stub that always returns return_value."""
    import src.server as srv
    # _is_out_of_scope(message, pipeline_module) — stub must accept both args.
    monkeypatch.setattr(srv, '_is_out_of_scope', lambda msg, mod: return_value)


def test_out_of_scope_queries_return_scope_response(monkeypatch):
    """Out-of-scope messages must get the scope-refusal response and skip RAG."""
    _patch_scope_classifier_returns(monkeypatch, True)
    server.SESSIONS.clear()

    with server.app.test_client() as client:
        for q in _OUT_OF_SCOPE_QUERIES:
            res = client.post('/api/chat', json={'message': q})
            data = res.get_json()
            assert data['needs_clarification'] is False, f"unexpected clarification for: {q}"
            assert data['groq_used'] is False, f"groq_used should be False for: {q}"
            assert data['verified_generics'] == [], f"no generics expected for: {q}"
            assert data['retrievals'] == [], f"no retrievals expected for: {q}"
            assert data.get('out_of_scope') is True, f"out_of_scope flag missing for: {q}"
            assert "outside my scope" in data['response'], f"wrong response for: {q}"
            assert "DoseWise" in data['response'], f"DoseWise branding missing for: {q}"


def test_out_of_scope_does_not_trigger_medication_pipeline(monkeypatch):
    """Ensure medication extraction and DDInter retrieval are NOT called for OOS messages."""
    _patch_scope_classifier_returns(monkeypatch, True)
    server.SESSIONS.clear()

    with server.app.test_client() as client:
        for q in _OUT_OF_SCOPE_QUERIES:
            res = client.post('/api/chat', json={'message': q})
            data = res.get_json()
            assert data['extracted'] == [], f"extracted should be empty for: {q}"
            assert data['resolved'] == [], f"resolved should be empty for: {q}"


def test_in_scope_medication_queries_are_not_blocked(monkeypatch):
    """Medication questions must NOT be blocked by the OOS guard (classifier returns False)."""
    _patch_scope_classifier_returns(monkeypatch, False)
    # Also stub generate_reply so no real Groq call is needed for the reply phase
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')
    server.SESSIONS.clear()

    with server.app.test_client() as client:
        for q in _IN_SCOPE_QUERIES:
            res = client.post('/api/chat', json={'message': q})
            data = res.get_json()
            assert data.get('out_of_scope') is not True, f"in-scope query blocked: {q}"


def test_conversational_greetings_are_not_blocked(monkeypatch):
    """Normal greetings must NOT be blocked by the OOS guard."""
    _patch_scope_classifier_returns(monkeypatch, False)
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'Hello!')
    server.SESSIONS.clear()

    with server.app.test_client() as client:
        for q in _CONVERSATIONAL_QUERIES:
            res = client.post('/api/chat', json={'message': q})
            data = res.get_json()
            assert data.get('out_of_scope') is not True, f"greeting blocked: {q}"
            assert data['needs_clarification'] is False, f"unexpected clarification for: {q}"


def test_safety_intent_takes_priority_over_oos(monkeypatch):
    """Safety messages must return the overdose warning even if OOS classifier would say YES."""
    # Even if _is_out_of_scope is stubbed to True, safety check runs first and short-circuits.
    _patch_scope_classifier_returns(monkeypatch, True)
    server.SESSIONS.clear()

    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'I took 5 Panadol tablets at once.'})
        data = res.get_json()
        assert "emergency or overdose" in data['response']
        assert data.get('out_of_scope') is not True


# ---------------------------------------------------------------------------
# Deterministic end-to-end regression tests (no monkeypatching / no Groq key)
# ---------------------------------------------------------------------------

def test_oos_ahmed_zewail_variants(monkeypatch):
    """'any info about Ahmed Zewail' and close variants must return the OOS response."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: None)
    server.SESSIONS.clear()
    queries = [
        'any info about Ahmed Zewail',
        'Who is Ahmed Zewail?',
        'Tell me about Ahmed Zewail',
    ]
    with server.app.test_client() as client:
        for q in queries:
            res = client.post('/api/chat', json={'message': q})
            data = res.get_json()
            assert data.get('out_of_scope') is True, f'Expected OOS for: {q}'
            assert 'outside my scope' in data['response'], f'Wrong response for: {q}'
            assert data['verified_generics'] == [], f'No generics expected for: {q}'
            assert data['retrievals'] == [], f'No retrievals expected for: {q}'


def test_oos_python_question(monkeypatch):
    """'Write me a Python program' must be blocked as OOS."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: None)
    server.SESSIONS.clear()
    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'Write me a Python program'})
        data = res.get_json()
        assert data.get('out_of_scope') is True
        assert 'outside my scope' in data['response']


def test_oos_capital_of_egypt(monkeypatch):
    """'What is the capital of Egypt?' must be blocked as OOS."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: None)
    server.SESSIONS.clear()
    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': "What's the capital of Egypt?"})
        data = res.get_json()
        assert data.get('out_of_scope') is True
        assert 'outside my scope' in data['response']


def test_greeting_hello_is_conversational(monkeypatch):
    """'Hello' must reach the conversational path, not be blocked as OOS."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'Hi there!')
    server.SESSIONS.clear()
    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'Hello'})
        data = res.get_json()
        assert data.get('out_of_scope') is not True
        assert data['needs_clarification'] is False
        assert data['response'] == 'Hi there!'


def test_panadol_amoxicillin_reaches_rag(monkeypatch):
    """'Can I take Panadol with amoxicillin?' must reach the RAG pipeline."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')
    server.SESSIONS.clear()
    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'Can I take Panadol with amoxicillin?'})
        data = res.get_json()
        assert data.get('out_of_scope') is not True
        assert set(data['verified_generics']) >= {'acetaminophen', 'amoxicillin'}
        assert len(data['retrievals']) >= 1


def test_simvastatin_followup_reaches_rag(monkeypatch):
    """'What about simvastatin?' (follow-up) must reach the RAG pipeline."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')
    server.SESSIONS.clear()
    session_id = 'oos-followup-test'
    with server.app.test_client() as client:
        # Seed the session with Panadol + amoxicillin first
        client.post('/api/chat', json={
            'session_id': session_id,
            'message': 'Can I take Panadol with amoxicillin?',
        })
        # Now ask the follow-up
        res = client.post('/api/chat', json={
            'session_id': session_id,
            'message': 'What about simvastatin?',
        })
        data = res.get_json()
        assert data.get('out_of_scope') is not True
        assert 'simvastatin' in data['verified_generics']
        assert any(
            frozenset((r['drug_a'].lower(), r['drug_b'].lower()))
            in {frozenset(('acetaminophen', 'simvastatin')), frozenset(('amoxicillin', 'simvastatin'))}
            for r in data['retrievals']
        ), 'Expected simvastatin interaction pair in retrievals'


# ---------------------------------------------------------------------------
# Exact regression tests as specified in requirements
# ---------------------------------------------------------------------------

def test_exact_oos_amed_zewail(monkeypatch):
    """Exact input 'any info about amed zewail' must return deterministic OOS response.

    Assertions (as specified):
    - response is OUT_OF_SCOPE (contains 'outside my scope')
    - response does NOT contain 'I am here to help'
    - no RAG retrieval occurs (retrievals == [])
    - Groq is NOT called (groq_used == False)
    """
    groq_called = []

    def track_groq(msg, res, hist):
        groq_called.append(msg)
        return 'should not reach Groq'

    monkeypatch.setattr(server, 'generate_reply', track_groq)
    server.SESSIONS.clear()

    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'any info about amed zewail'})
        data = res.get_json()

    assert data.get('out_of_scope') is True, \
        f"Expected out_of_scope=True, got response: {data.get('response')!r}"
    assert 'outside my scope' in data['response'], \
        f"Response must contain 'outside my scope', got: {data['response']!r}"
    assert 'I am here to help' not in data['response'], \
        "Response must NOT contain the conversational fallback"
    assert data['retrievals'] == [], \
        f"No RAG retrieval expected, got: {data['retrievals']}"
    assert data['groq_used'] is False, \
        "Groq must NOT be called for OOS messages"
    assert groq_called == [], \
        f"generate_reply was unexpectedly called with: {groq_called}"


def test_exact_hi_ranya_is_conversational(monkeypatch):
    """'Hi, my name is Ranya' must reach Groq conversational path — never OOS."""
    groq_called = []

    def track_groq(msg, res, hist):
        groq_called.append(msg)
        return 'Hello Ranya!'

    monkeypatch.setattr(server, 'generate_reply', track_groq)
    server.SESSIONS.clear()

    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'Hi, my name is Ranya'})
        data = res.get_json()

    assert data.get('out_of_scope') is not True, \
        "Greeting must NOT be classified as out-of-scope"
    assert data['needs_clarification'] is False
    assert data['groq_used'] is True, "Groq must be called for greetings"
    assert groq_called == ['Hi, my name is Ranya'], \
        f"generate_reply not called as expected, got: {groq_called}"


def test_exact_panadol_amoxicillin_rag(monkeypatch):
    """'Can I take Panadol with amoxicillin?' must run RAG + DDInter, never OOS."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')
    server.SESSIONS.clear()

    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'Can I take Panadol with amoxicillin?'})
        data = res.get_json()

    assert data.get('out_of_scope') is not True
    assert set(data['verified_generics']) >= {'acetaminophen', 'amoxicillin'}, \
        f"Expected both generics, got: {data['verified_generics']}"
    assert len(data['retrievals']) >= 1, \
        "Expected at least one DDInter retrieval"
    assert data['needs_clarification'] is False


def test_exact_simvastatin_followup_rag(monkeypatch):
    """'What about simvastatin?' follow-up must run RAG using previous session meds."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')
    server.SESSIONS.clear()
    session_id = 'exact-followup-zewail-test'

    with server.app.test_client() as client:
        # Prime the session
        client.post('/api/chat', json={
            'session_id': session_id,
            'message': 'Can I take Panadol with amoxicillin?',
        })
        # Follow-up
        res = client.post('/api/chat', json={
            'session_id': session_id,
            'message': 'What about simvastatin?',
        })
        data = res.get_json()

    assert data.get('out_of_scope') is not True
    assert 'simvastatin' in data['verified_generics']
    pairs = {frozenset((r['drug_a'].lower(), r['drug_b'].lower())) for r in data['retrievals']}
    assert frozenset(('acetaminophen', 'simvastatin')) in pairs or \
           frozenset(('amoxicillin', 'simvastatin')) in pairs, \
        f"Expected simvastatin pair in retrievals, got pairs: {pairs}"


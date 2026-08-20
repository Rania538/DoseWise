from src import rag_pipeline as rp


def test_empty_input_has_no_medication_or_interaction_fallback():
    result = rp.process_message('')
    assert result['extracted'] == []
    assert result['verified_generics'] == []
    assert result['retrievals'] == []
    assert 'interaction' not in result['response'].lower() or 'database' in result['response'].lower()


def test_unknown_medication_stops_before_retrieval():
    result = rp.process_message('XyzUnknown and amoxicillin')
    assert any(not item['verified'] for item in result['resolved'])
    assert result['retrievals'] == []
    assert 'Did you mean' in result['response']


def test_confirmation_and_rejection_keep_conversation_safe():
    context = {}
    first = rp.process_message('XyzUnknown and amoxicillin', conversation_context=context)
    assert 'pending_clarification' in context

    confirmed = rp.process_message('yes', conversation_context=context)
    assert all(item.get('input', '').lower() != 'yes' for item in confirmed['resolved'])
    assert 'pending_clarification' not in context

    context = {}
    rp.process_message('XyzUnknown and amoxicillin', conversation_context=context)
    rejected = rp.process_message('no', conversation_context=context)
    assert rejected['retrievals'] == []
    assert 'correct medication name' in rejected['response']


def test_conversational_arabic_words_are_not_medications():
    result = rp.process_message('أنا كنت عند دكتور وكتبلي بنادول، ينفع أخده؟')
    extracted = {rp.normalize_text(item) for item in result['extracted']}
    assert 'بنادول' in extracted
    assert not extracted.intersection({'أنا', 'دكتور', 'كتبلي', 'دواء', 'ينفع'})


def test_full_sentence_and_multiple_medications_regression():
    result = rp.process_message(
        'I visited a neurologist and they prescribed Panadol. '
        'Later my ENT doctor prescribed amoxicillin. Can I take them together?'
    )
    message_norm = rp.normalize_text(result['language'] and
        'I visited a neurologist and they prescribed Panadol. '
        'Later my ENT doctor prescribed amoxicillin. Can I take them together?')
    assert len(result['verified_generics']) == 2
    assert not any(rp.normalize_text(item['input']) == message_norm for item in result['resolved'])
    assert len(result['retrievals']) == 1


def test_missing_interaction_evidence_is_explicit_and_grounded():
    result = rp.generate_response(
        'amoxicillin and naltrexone',
        'en',
        [],
        [{
            'drug_a': 'amoxicillin',
            'drug_b': 'naltrexone',
            'interaction_level': None,
            'source': 'DDInter',
            'ddinter_rows': 0,
            'matches': [],
            'chunk_ids': [],
            'source_rows': [],
        }],
    )
    assert 'No interaction was identified in the available DDInter knowledge base' in result
    assert 'guaranteed' not in result.lower()


def test_retrieved_evidence_has_stable_audit_metadata():
    result = rp.process_message('Panadol with amoxicillin')
    matched = next(item for item in result['retrievals'] if item['ddinter_rows'])
    record = matched['matches'][0]
    assert matched['source'] == 'DDInter'
    assert matched['chunk_ids'] == [record['chunk_id']]
    assert record['chunk_id'] == f"ddinter-row-{record['source_row']}"
    assert record['evidence'].endswith(f"row {record['source_row']}")

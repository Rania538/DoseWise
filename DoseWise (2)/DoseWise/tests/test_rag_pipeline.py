import pprint

import pytest

from src import rag_pipeline as rp


def show(result):
    pprint.pprint(result)


def run_case(message):
    out = rp.process_message(message)
    show({
        'message': message,
        'language': out['language'],
        'extracted': out['extracted'],
        'resolved': [{
            'input': r['input'],
            'normalized_input': r['normalized_input'],
            'generic_name': r.get('generic_name'),
            'verified': r.get('verified'),
            'match_type': r.get('match_type'),
        } for r in out['resolved']],
        'verified_generics': out['verified_generics'],
        'retrievals': out['retrievals'],
        'response': out['response'],
    })
    return out


def test_panadol_amoxicillin():
    out = run_case('I was prescribed Panadol and amoxicillin. Can I take them together?')
    assert 'panadol' in [s.lower() for s in out['extracted']] or 'Panadol' in out['extracted']
    # both should resolve to verified generics
    assert any(r['verified'] for r in out['resolved'])
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]
    # retrieval should include acetaminophen+amoxicillin
    pairs = out['retrievals']
    assert any(('acetaminophen' in p['drug_a'].lower() or 'acetaminophen' in p['drug_b'].lower()) for p in pairs)


def test_benadol_arabic_plus_amoxicillin():
    msg = 'أنا خدت بنادول و amoxicillin، ينفع آخدهم مع بعض؟'
    out = run_case(msg)
    assert out['language'] == 'ar'
    assert any('بنادول' in e for e in out['extracted'] or []) or any('بنادول' in r['input'] for r in out['resolved'])
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]


def test_pandol_misspell_amoxicilin():
    out = run_case('Pandol and amoxicilin')
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]
    assert 'amoxicillin' in [g.lower() for g in out['verified_generics']]


def test_paracetamol_amoxicillin():
    out = run_case('Paracetamol + amoxicillin')
    assert 'paracetamol' in [e.lower() for e in out['extracted']] or 'paracetamol' in [r['input'].lower() for r in out['resolved']]
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]


def test_unknown_plus_amoxicillin():
    out = run_case('XyzUnknown and amoxicillin')
    # unknown should be ambiguous and pipeline should ask for clarification
    assert any(not r['verified'] for r in out['resolved'])
    assert ('Did you mean' in out['response'] or 'هل كنت تقصد' in out['response'])


def test_panadol_amoxicillin_deduplication():
    # Panadol + amoxicillin -> exactly one retrieval pair (acetaminophen + amoxicillin).
    out = run_case('I was prescribed Panadol and amoxicillin. Can I take them together?')
    assert len(out['verified_generics']) == 2
    assert len(out['retrievals']) == 1
    pair = out['retrievals'][0]
    drugs = {pair['drug_a'].lower(), pair['drug_b'].lower()}
    assert 'acetaminophen' in drugs
    assert 'amoxicillin' in drugs


def test_benadol_acetaminophen_deduplication():
    # بنادول + acetaminophen -> exactly one medication: acetaminophen.
    out = run_case('هل ينفع آخد بنادول مع acetaminophen؟')
    assert len(out['verified_generics']) == 1
    assert out['verified_generics'][0].lower() == 'acetaminophen'
    assert len(out['retrievals']) == 0


def test_parse_medication_followup():
    assert rp.parse_medication_followup("طب و simvastatin؟") == (True, ["simvastatin"])
    assert rp.parse_medication_followup("طيب و simvastatin؟") == (True, ["simvastatin"])
    assert rp.parse_medication_followup("و simvastatin؟") == (True, ["simvastatin"])
    assert rp.parse_medication_followup("وsimvastatin؟") == (True, ["simvastatin"])
    assert rp.parse_medication_followup("ماذا عن simvastatin؟") == (True, ["simvastatin"])
    assert rp.parse_medication_followup("what about simvastatin?") == (True, ["simvastatin"])
    assert rp.parse_medication_followup("how about simvastatin?") == (True, ["simvastatin"])
    assert rp.parse_medication_followup("and simvastatin?") == (True, ["simvastatin"])
    assert rp.parse_medication_followup("طب إنت بتعمل إيه؟") == (False, [])
    assert rp.parse_medication_followup("طب أعمل إيه؟") == (False, [])
    assert rp.parse_medication_followup("طب") == (False, [])


def test_followup_retrieval_logic():
    out = rp.process_message("طب و simvastatin؟", active_medications=["acetaminophen", "amoxicillin"])
    # 2 new pairs
    assert len(out['retrievals']) == 2
    drugs = set()
    for pair in out['retrievals']:
        drugs.add(pair['drug_a'].lower())
        drugs.add(pair['drug_b'].lower())
    assert 'acetaminophen' in drugs
    assert 'amoxicillin' in drugs
    assert 'simvastatin' in drugs
    assert len(out['verified_generics']) == 1
    assert out['verified_generics'][0] == 'simvastatin'


def test_normal_conversation_not_medication():
    # Pure conversational messages shouldn't produce verified generics.
    # Note: process_message is the RAG pipeline alone, so it tries to extract regardless,
    # but the parser should not find verified meds for these.
    out = rp.process_message("Hi, how are you?")
    assert len(out['verified_generics']) == 0
    out = rp.process_message("طب")
    assert len(out['verified_generics']) == 0


def test_followup_no_context():
    # A follow-up without active meds should just extract the med but not generate retrievals
    # (retrievals need pairs)
    out = rp.process_message("what about simvastatin?")
    assert 'simvastatin' in out['verified_generics']
    assert len(out['retrievals']) == 0


def test_three_medications_pairs():
    out = run_case('Panadol, amoxicillin and simvastatin')
    # three meds -> 3 unique pairs
    assert len(out['verified_generics']) >= 2
    # retrievals should include pair evidence for each unique pair (may be empty matches but pairs generated)
    assert len(out['retrievals']) == (len(out['verified_generics']) * (len(out['verified_generics']) - 1)) // 2


def test_english_natural_question():
    out = run_case('I was prescribed Panadol by my GP and amoxicillin by the ENT. Can I take them together?')
    assert out['language'] == 'en'
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]


def test_arabizi_natural_question():
    msg = 'ana kont 3and doctor w katbly Panadol w doktor tany katbly amoxicillin, ynf3 akhodhom ma ba3d?'
    out = run_case(msg)
    # detect as Arabic (arabizi)
    assert out['language'] == 'ar'
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]


def test_exact_extract_panadol_amoxicillin():
    msg = 'Can I take Panadol with amoxicillin?'
    out = run_case(msg)
    # extraction must find exactly the two medication tokens
    extracted_norm = [s.lower() for s in out['extracted']]
    assert 'panadol' in extracted_norm and 'amoxicillin' in extracted_norm
    assert len(extracted_norm) == 2
    # both should resolve to verified generics
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]
    assert 'amoxicillin' in [g.lower() for g in out['verified_generics']]


def test_variants_and_confirmation_flow():
    # variant without question mark
    out = run_case('Can I take Panadol with amoxicillin')
    assert 'panadol' in [s.lower() for s in out['extracted']]

    # arabic phrasing
    out = run_case('بنادول مع amoxicillin ينفع؟')
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]

    # arabizi
    out = run_case('ana 5dt Pandol w amoxicilin, ynf3 akhodhom m3 ba3d?')
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]

    # plus sign variant
    out = run_case('Panadol + amoxicillin')
    assert 'acetaminophen' in [g.lower() for g in out['verified_generics']]

    # unknown then clarification -> yes (use the 'XyzUnknown and amoxicillin' phrasing)
    ctx = {}
    first = run_case('XyzUnknown and amoxicillin')
    # should ask for clarification about the unknown
    assert any(not r['verified'] for r in first['resolved'])
    assert ('Did you mean' in first['response'] or 'هل كنت تقصد' in first['response'])
    # store pending clarification from pipeline (simulate the context lifecycle)
    # rerun using process_message directly to exercise context-aware confirmation
    import src.rag_pipeline as rp
    ctx = {}
    res1 = rp.process_message('XyzUnknown and amoxicillin', conversation_context=ctx)
    assert 'pending_clarification' in ctx
    res2 = rp.process_message('yes', conversation_context=ctx)
    # after confirming, 'yes' should not be interpreted as a medication
    assert all('yes' not in (r.get('input') or '').lower() for r in res2['resolved'])

    # no after clarification should trigger request for correct name
    ctx2 = {}
    _ = rp.process_message('XyzUnknown and amoxicillin', conversation_context=ctx2)
    assert 'pending_clarification' in ctx2
    resp_no = rp.process_message('no', conversation_context=ctx2)
    assert 'Please provide the correct medication name' in resp_no['response'] or 'من فضلك' in resp_no['response']

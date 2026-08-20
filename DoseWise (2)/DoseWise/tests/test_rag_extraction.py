import pprint
import pytest

from src.rag_pipeline import process_message
from src.drug_resolver import normalize_text


def show(o):
    pprint.pprint(o)


@pytest.mark.parametrize('message, expected_meds, expect_clarify', [
    ("Hello, my name is Ranya. What can you help me with?", [], False),
    ("Panadol + amoxicillin", ['panadol', 'amoxicillin'], False),
    ("Can I take Panadol with amoxicillin?", ['panadol', 'amoxicillin'], False),
    ("انا باخد بانادول مع amoxicillin", ['بنادول', 'amoxicillin'], False),
    ("أنا كنت عند دكتور المخ والأعصاب وكتبلي Panadol وبعدها رحت لدكتور الأنف والأذن وكتبلي amoxicillin، ينفع أخد الاتنين مع بعض؟",
     ['panadol', 'amoxicillin'], False),
    ("I visited a neurologist and they prescribed Panadol. Later my ENT doctor prescribed amoxicillin. Can I take them together?",
     ['panadol', 'amoxicillin'], False),
    ("ana kont 3and doctor w katbly Pandol w doctor tany katbly amoxicilin, ynf3 akhodhom m3 ba3d?",
     ['pandol', 'amoxicilin'], False),
    ("I have a headache and took Panadol.", ['panadol'], False),
    ("I took something called XyzUnknown.", ['XyzUnknown'], True),
])
def test_extraction_and_resolution(message, expected_meds, expect_clarify):
    out = process_message(message)
    # Print structured outputs for inspection
    result = {
        'message': message,
        'extracted': out['extracted'],
        'resolved_inputs': [r['input'] for r in out['resolved']],
        'verified_generics': out['verified_generics'],
        'retrievals': out['retrievals'],
        'response': out['response'],
    }
    show(result)

    # Ensure expected meds appear in extracted (case-insensitive)
    extracted_norm = [normalize_text(x) for x in out['extracted']]
    for med in expected_meds:
        assert normalize_text(med) in extracted_norm

    # Ensure we did not send the entire long message to resolver
    msg_norm = normalize_text(message)
    assert not any(normalize_text(r['input']) == msg_norm for r in out['resolved'])

    # Verify verified generics for known meds
    if 'panadol' in [m.lower() for m in expected_meds] or 'pandol' in [m.lower() for m in expected_meds] or 'بنادول' in [m.lower() for m in expected_meds]:
        assert any('acetaminophen' in (g.lower()) for g in out['verified_generics'])
    if any('amoxicil' in m.lower() or 'amoxicillin' in m.lower() for m in expected_meds):
        assert any('amoxicillin' in (g.lower()) for g in out['verified_generics'])

    # Clarification expectation
    ambiguous = any(not r.get('verified') for r in out['resolved'])
    assert ambiguous == expect_clarify

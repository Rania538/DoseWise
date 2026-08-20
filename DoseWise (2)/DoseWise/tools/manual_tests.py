import json
import sys
sys.path.insert(0, '.')
from src.rag_pipeline import process_message

CASES = {
    'arabic': "أنا كنت عند دكتور المخ والأعصاب وكتبلي بنادول وبعدها رحت لدكتور الأنف والأذن وكتبلي amoxicillin، ينفع أخد الاتنين مع بعض؟",
    'english': "I visited a neurologist and they prescribed Panadol. Later my ENT doctor prescribed amoxicillin. Can I take them together?",
    'arabizi': "ana kont 3and doctor w katbly Pandol w doctor tany katbly amoxicilin, ynf3 akhodhom m3 ba3d?",
    'unknown': 'I took something called XyzUnknown.',
    'ambiguous': 'I was prescribed something called "PanadolX" and amoxicillin',
    'three': 'Panadol, amoxicillin and simvastatin',
    'empty': ''
}

results = {}
for k,m in CASES.items():
    try:
        out = process_message(m)
        results[k] = {
            'message': m,
            'language': out['language'],
            'extracted': out['extracted'],
            'resolved': [{'input': r.get('input'), 'generic_name': r.get('generic_name'), 'verified': r.get('verified'), 'source': r.get('source'), 'reason': r.get('reason')} for r in out['resolved']],
            'verified_generics': out['verified_generics'],
            'retrievals': out['retrievals'],
            'response': out['response']
        }
    except Exception as e:
        results[k] = {'error': str(e)}

print(json.dumps(results, ensure_ascii=False, indent=2))

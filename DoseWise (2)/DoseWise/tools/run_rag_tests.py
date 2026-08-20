import json
import sys
from pprint import pprint

sys.path.insert(0, '.')
from src.rag_pipeline import process_message


cases = [
    'I was prescribed Panadol and amoxicillin. Can I take them together?',
    'أنا خدت بنادول و amoxicillin، ينفع آخدهم مع بعض؟',
    'Pandol and amoxicilin',
    'Paracetamol + amoxicillin',
    'XyzUnknown and amoxicillin',
    'Panadol, amoxicillin and simvastatin',
    'أنا كنت عند دكتور المخ والأعصاب وكتبلي دواء، وبعدها رحت لدكتور الأنف والأذن وكتبلي دواء تاني، ينفع أخد الاتنين مع بعض؟',
    'I was prescribed Panadol by my GP and amoxicillin by the ENT. Can I take them together?',
    'ana kont 3and doctor w katbly Panadol w doktor tany katbly amoxicillin, ynf3 akhodhom ma ba3d?'
]

results = []
for msg in cases:
    r = process_message(msg)
    simplified = {
        'message': msg,
        'language': r['language'],
        'extracted_medications': r['extracted'],
        'resolver_results': [
            {
                'input': d.get('input'),
                'normalized_input': d.get('normalized_input'),
                'generic_name': d.get('generic_name'),
                'active_ingredients': d.get('active_ingredients'),
                'verified': d.get('verified'),
                'match_type': d.get('match_type'),
                'source': d.get('source'),
                'reason': d.get('reason'),
            }
            for d in r['resolved']
        ],
        'verified_generics': r['verified_generics'],
        'ddinter_retrievals': [
            {
                'drug_a': p.get('drug_a'),
                'drug_b': p.get('drug_b'),
                'interaction_level': p.get('interaction_level'),
                'ddinter_rows': p.get('ddinter_rows'),
                'matches': p.get('matches'),
            }
            for p in r['retrievals']
        ],
        'final_response': r['response'],
    }
    results.append(simplified)

print(json.dumps(results, ensure_ascii=False, indent=2))

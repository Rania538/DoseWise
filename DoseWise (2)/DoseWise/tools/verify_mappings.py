import requests, json

to_check = [
    ('Can I take Panadol with amoxicillin?', ['acetaminophen','amoxicillin']),
    ('panadol and amoxicillin', ['acetaminophen','amoxicillin']),
    ('Pandol and amoxicillin', ['acetaminophen','amoxicillin']),
    ('بنادول و amoxicillin', ['acetaminophen','amoxicillin']),
    ('Paracetamol and amoxicillin', ['acetaminophen','amoxicillin']),
    ('Doliprane and amoxicillin', ['acetaminophen','amoxicillin']),
    ('amoxicillin and water', ['amoxicillin']),
    ('amoxicilin and water', ['amoxicillin']),
]

failures = []

for msg, expected_generics in to_check:
    r = requests.post('http://127.0.0.1:5000/api/chat', json={'message':msg})
    if r.status_code != 200:
        failures.append((msg, 'http_error', r.status_code))
        continue
    data = r.json()
    verified = data.get('verified_generics') or []
    # normalize lower
    vset = set([v.lower() for v in verified])
    exp_set = set([e.lower() for e in expected_generics])
    # ensure Panadol never maps to nadolol
    if 'panadol' in msg.lower():
        if any('nadolol' in v.lower() for v in verified):
            failures.append((msg, 'panadol_resolved_to_nadolol', verified))
            continue
    if not exp_set.issubset(vset):
        failures.append((msg, 'mismatch', {'expected': list(exp_set), 'found': list(vset)}))

# Unknown drugs test
r = requests.post('http://127.0.0.1:5000/api/chat', json={'message':'XyzUnknown and amoxicillin'})
if r.status_code==200:
    data = r.json()
    if not data.get('needs_clarification'):
        failures.append(('Unknown_should_clarify', 'no_clarification', data))
    # make sure resolver wasn't sent full sentence: resolved inputs should not include full sentence
    for ritem in data.get('resolved',[]):
        if ritem.get('input') and ritem.get('input').strip().lower() == 'xyzunknown and amoxicillin':
            failures.append(('full_sentence_sent', 'resolver_echo', ritem))

# Full sentence must not be sent
r = requests.post('http://127.0.0.1:5000/api/chat', json={'message':'Can I take Panadol with amoxicillin? I have a headache and fever.'})
if r.status_code==200:
    data = r.json()
    for ritem in data.get('resolved',[]):
        if ritem.get('input') and ritem.get('input').strip().lower().startswith('can i take'):
            failures.append(('full_sentence_sent_2', 'resolver_echo', ritem))

print('FAILURES:', json.dumps(failures, indent=2, ensure_ascii=False))
if not failures:
    print('All mapping verifications passed')

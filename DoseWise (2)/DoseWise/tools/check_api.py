import requests, json
r = requests.post('http://127.0.0.1:5000/api/chat', json={'message':'XyzUnknown and amoxicillin'})
print('status', r.status_code)
try:
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print('no json', e, r.text)

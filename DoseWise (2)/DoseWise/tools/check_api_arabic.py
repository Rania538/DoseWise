import requests, json
msg = 'أنا كنت عند دكتور وكتبلي Panadol وبعدها دكتور تاني كتبلي amoxicillin، ينفع أخد الاتنين؟'
r = requests.post('http://127.0.0.1:5000/api/chat', json={'message':msg})
print('status', r.status_code)
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

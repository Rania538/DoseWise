import json
import urllib.request

URL = 'http://127.0.0.1:5000/api/chat'
CASES = [
    'Can I take Panadol with amoxicillin?',
    'أنا كنت عند دكتور وكتبلي Panadol وبعدها دكتور تاني كتبلي amoxicillin، ينفع أخد الاتنين؟',
    'ana 5dt Pandol w amoxicilin, ynf3 akhodhom m3 ba3d?',
    'XyzUnknown and amoxicillin',
    'Panadol, amoxicillin and simvastatin'
]

def post(msg):
    data = json.dumps({'message': msg}).encode('utf-8')
    req = urllib.request.Request(URL, data=data, headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            print('===', msg)
            print(r.read().decode('utf-8'))
    except Exception as e:
        print('ERROR', e)

if __name__=='__main__':
    for c in CASES:
        post(c)

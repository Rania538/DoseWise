import sys, json, urllib.request
url='http://127.0.0.1:5000/api/chat'
prompts=[
  'Can I take Panadol with amoxicillin?',
  'أنا كنت عند دكتور المخ والأعصاب وكتبلي بنادول وبعدها رحت لدكتور الأنف والأذن وكتبلي amoxicillin، ينفع أخد الاتنين مع بعض؟',
  'ana 5dt Pandol w amoxicilin, ynf3 akhodhom m3 ba3d?',
  'I took XyzUnknown with amoxicillin.',
  'Panadol, amoxicillin and simvastatin'
]
for p in prompts:
  data=json.dumps({'message':p}).encode('utf-8')
  req=urllib.request.Request(url,data,headers={'Content-Type':'application/json'})
  try:
    resp=urllib.request.urlopen(req,timeout=10)
    print('PROMPT:',p)
    print(resp.read().decode('utf-8'))
  except Exception as e:
    print('ERROR for prompt',p, e)

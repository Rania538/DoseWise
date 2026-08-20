import sys
sys.path.insert(0, 'src')
from rag_pipeline import extract_medications, process_message, normalize_text

def diag(msg):
    print('MSG:', msg)
    print('NORMALIZED:', normalize_text(msg))
    print('EXTRACTED:', extract_medications(msg))
    out = process_message(msg)
    print('PROCESS_VERIFIED_GENERICS:', out['verified_generics'])
    print('PROCESS_RESOLVED:', out['resolved'])

if __name__=='__main__':
    diag('XyzUnknown and amoxicillin')
    diag('XyzUnknown and Panadol')
    diag('XyzUnknown')

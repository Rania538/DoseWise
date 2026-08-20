from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time, os

BASE = os.environ.get('BASE_URL','http://127.0.0.1:5000')
SCREEN_DIR = 'tools/selenium_screens'

TESTS = [
    {'name':'english_simple','msg':'Can I take Panadol with amoxicillin?','rtl':False,'expect_detected_pair':('panadol','acetaminophen'),'clarify':False,'expect_count_meds':2},
    {'name':'arabic','msg':'أنا كنت عند دكتور وكتبلي Panadol وبعدها دكتور تاني كتبلي amoxicillin، ينفع أخد الاتنين؟','rtl':True,'expect_detected_pair':('panadol','acetaminophen'),'clarify':False,'expect_count_meds':2},
    {'name':'arabizi','msg':'ana 5dt Pandol w amoxicilin, ynf3 akhodhom m3 ba3d?','rtl':True,'expect_detected_pair':('pandol','acetaminophen'),'clarify':False,'expect_count_meds':2},
    {'name':'three_meds','msg':'Panadol, amoxicillin and simvastatin','rtl':False,'expect_count_meds':3,'clarify':False},
    {'name':'unknown_clarify','msg':'XyzUnknown and amoxicillin','rtl':False,'clarify':True}
]

os.makedirs(SCREEN_DIR, exist_ok=True)

opts = Options()
opts.headless = True
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-dev-shm-usage')
opts.add_argument('--window-size=1200,1600')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
driver.get(BASE)

# wait for health ready or short timeout
time.sleep(1)
try:
    for i in range(10):
        h = driver.find_element(By.ID,'health').text
        if 'Backend: ready' in h: break
        time.sleep(0.5)
except Exception:
    pass

for t in TESTS:
    name = t['name']
    print('Running', name)
    input_el = driver.find_element(By.ID,'messageInput')
    send_btn = driver.find_element(By.ID,'sendBtn')
    # clear previous chat to avoid carryover between tests
    try:
        clear_btn = driver.find_element(By.ID,'clearBtn')
        clear_btn.click()
        time.sleep(0.2)
    except Exception:
        pass
    input_el.clear(); input_el.send_keys(t['msg'])
    send_btn.click()

    # wait for typing indicator removal and assistant message
    timeout = time.time() + 15
    while time.time() < timeout:
        try:
            typing = driver.find_elements(By.ID,'typing')
            if not typing:
                # at least one assistant content exists
                elems = driver.find_elements(By.CSS_SELECTOR,'.msg.assistant .content')
                if elems:
                    break
        except Exception:
            pass
        time.sleep(0.3)

    # screenshot chat area
    chat = driver.find_element(By.ID,'chatArea')
    chat_png = f"{SCREEN_DIR}/{name}.png"
    chat.screenshot(chat_png)

    # verify last user message equals sent text
    users = driver.find_elements(By.CSS_SELECTOR,'.msg.user .content')
    if not users:
        raise RuntimeError('No user messages found')
    last_user = users[-1].text.strip()
    if last_user != t['msg']:
        raise RuntimeError(f'User message mismatch: expected "{t["msg"]}", got "{last_user}"')

    # clarif
    if t.get('clarify'):
        # ensure assistant asked clarification (text contains Did you mean or Arabic question mark)
        assists = [a.text for a in driver.find_elements(By.CSS_SELECTOR,'.msg.assistant .content')]
        if not any('Did you mean' in s or '؟' in s for s in assists[-3:]):
            raise RuntimeError('Clarification not shown when expected')
        # ensure no detected meds card present
        if driver.find_elements(By.CSS_SELECTOR,'.card .pair'):
            raise RuntimeError('Detected meds card present while clarification expected')
        continue

    # detected meds rows
    rows = [r.text.strip() for r in driver.find_elements(By.CSS_SELECTOR,'.card .interaction .meta-line')]
    if t.get('expect_count_meds'):
        if len(rows) < t['expect_count_meds']:
            raise RuntimeError(f'Expected {t["expect_count_meds"]} detected meds rows, found {len(rows)}')
    elif t.get('expect_detected_pair'):
        ee, ev = t['expect_detected_pair']
        found = any(ee in r.lower() and ev in r.lower() for r in rows)
        if not found:
            raise RuntimeError(f'Expected mapping {ee}->{ev} not found in rows {rows}')

    # interaction cards
    interactions = driver.find_elements(By.CSS_SELECTOR,'.card.interaction')
    if t.get('expect_count_meds') and len(interactions)==0:
        raise RuntimeError('Expected interaction cards but none found')

    # RTL check
    if t.get('rtl'):
        assists = driver.find_elements(By.CSS_SELECTOR,'.msg.assistant')
        if not assists: raise RuntimeError('No assistant messages for RTL check')
        last = assists[-1]
        dir_attr = last.get_attribute('dir')
        class_attr = last.get_attribute('class')
        if dir_attr != 'rtl' and 'rtl' not in (class_attr or ''):
            raise RuntimeError('Expected RTL but message not marked RTL')

print('Selenium UI tests passed; screenshots at', SCREEN_DIR)

driver.quit()
print('done')

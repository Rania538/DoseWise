from playwright.sync_api import sync_playwright
import time
import os

BASE = os.environ.get('BASE_URL', 'http://127.0.0.1:5000')
SCREEN_DIR = 'tools/playwright_screens'

TESTS = [
    {
        'name': 'english_simple',
        'msg': 'Can I take Panadol with amoxicillin?',
        'expect_detected_pair': ('panadol','acetaminophen'),
        'expect_interaction': True,
        'rtl': False
    },
    {
        'name': 'arabic',
        'msg': 'أنا كنت عند دكتور وكتبلي Panadol وبعدها دكتور تاني كتبلي amoxicillin، ينفع أخد الاتنين؟',
        'expect_detected_pair': ('panadol','acetaminophen'),
        'expect_interaction': True,
        'rtl': True
    },
    {
        'name': 'arabizi',
        'msg': 'ana 5dt Pandol w amoxicilin, ynf3 akhodhom m3 ba3d?',
        'expect_detected_pair': ('pandol','acetaminophen'),
        'expect_interaction': True,
        'rtl': True
    },
    {
        'name': 'three_meds',
        'msg': 'Panadol, amoxicillin and simvastatin',
        'expect_count_meds': 3,
        'expect_interaction': True,
        'rtl': False
    },
    {
        'name': 'unknown_clarify',
        'msg': 'XyzUnknown and amoxicillin',
        'expect_needs_clarification': True,
        'rtl': False
    }
]

os.makedirs(SCREEN_DIR, exist_ok=True)

def wait_for_response(page, timeout=10000):
    # wait until typing indicator removed and an assistant message appears
    page.wait_for_selector('#typing', state='detached', timeout=timeout)
    # wait for at least one assistant content block
    el = page.wait_for_selector('.msg.assistant .content', timeout=timeout)
    return el

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    page.goto(BASE)
    time.sleep(0.5)

    # ensure health is ready
    try:
        page.wait_for_selector('#health:text("Backend: ready")', timeout=10000)
    except Exception:
        print('Warning: backend health not ready; proceeding')

    # helper to send message
    def send(msg):
        page.fill('#messageInput', msg)
        page.click('#sendBtn')

    sent_user_count = 0

    for t in TESTS:
        name = t['name']
        print('Running', name)
        send(t['msg'])
        sent_user_count += 1
        # wait for response
        try:
            wait_for_response(page, timeout=15000)
        except Exception as e:
            raise RuntimeError(f'No response for {name}: {e}')

        # take screenshot of chat area
        chat = page.query_selector('#chatArea')
        chat.screenshot(path=f"{SCREEN_DIR}/{name}.png")

        # assertions
        # check last user message equals sent text (no duplicates)
        users = page.query_selector_all('.msg.user .content')
        if not users:
            raise RuntimeError('No user messages found after send')
        last_user = users[-1].inner_text().strip()
        if last_user != t['msg']:
            raise RuntimeError(f'Last user message mismatch for {name}: expected "{t["msg"]}", found "{last_user}"')

        # check clarification behavior
        if t.get('expect_needs_clarification'):
            # ensure assistant message contains clarifying prompt and there is NO detected meds card
            assistant_txts = [a.inner_text() for a in page.query_selector_all('.msg.assistant .content')]
            if not any('Did you mean' in s or '؟' in s for s in assistant_txts[-3:]):
                raise RuntimeError('Clarification not shown for unknown candidate')
            if page.query_selector('.card .pair'):
                # detected meds card present -> failure
                raise RuntimeError('Detected meds card present when clarification expected')
            continue

        # check detected meds card
        detected = page.query_selector('.card .pair')
        if not detected:
            raise RuntimeError('No detected meds card shown')
        # read all detected lines
        rows = [r.inner_text().strip() for r in page.query_selector_all('.card .interaction .meta-line')]
        if t.get('expect_count_meds'):
            if len(rows) < t['expect_count_meds']:
                raise RuntimeError(f'Expected {t["expect_count_meds"]} detected meds rows, found {len(rows)}')
        else:
            # check specific mapping exists
            exp_e, exp_v = t['expect_detected_pair']
            found_pair = any(exp_e in r.lower() and exp_v in r.lower() for r in rows)
            if not found_pair:
                raise RuntimeError(f'Expected mapping {exp_e} -> {exp_v} not found in detected rows: {rows}')

        # check interaction cards exist for interaction retrievals
        interactions = page.query_selector_all('.card.interaction')
        if t.get('expect_interaction') and len(interactions) == 0:
            raise RuntimeError('Expected interaction cards but none rendered')

        # check RTL if expected: last assistant message wrapper should have dir=rtl or class rtl
        if t.get('rtl'):
            assist = page.query_selector_all('.msg.assistant')[-1]
            dir_attr = assist.get_attribute('dir')
            class_attr = assist.get_attribute('class')
            if dir_attr != 'rtl' and 'rtl' not in (class_attr or ''):
                raise RuntimeError('Expected assistant message to be RTL but it is not')

    print('All Playwright UI tests passed. Screenshots in', SCREEN_DIR)
    browser.close()

print('done')

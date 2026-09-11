import json
import time
import pytest
from src import server


@pytest.fixture(autouse=True)
def clean_server_state():
    """Ensure clean session and rate limiter state before every test."""
    server.SESSIONS.clear()
    server._reset_rate_limits()
    yield
    server.SESSIONS.clear()
    server._reset_rate_limits()


# ===========================================================================
# 1. Session Ownership & Isolation Tests
# ===========================================================================

def test_client_cannot_access_another_clients_session(monkeypatch):
    """Client B must not be allowed to access Client A's existing session."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')

    session_id = 'client-a-confidential-session'

    # Client A creates the session and seeds active medications
    with server.app.test_client() as client_a:
        res_a = client_a.post('/api/chat', json={
            'session_id': session_id,
            'message': 'Can I take Panadol with amoxicillin?'
        })
        assert res_a.status_code == 200
        data_a = res_a.get_json()
        assert set(data_a['verified_generics']) >= {'acetaminophen', 'amoxicillin'}

    # Client B (distinct client / different cookies) attempts to use Client A's session ID
    with server.app.test_client() as client_b:
        res_b = client_b.post('/api/chat', json={
            'session_id': session_id,
            'message': 'What about simvastatin?'
        })
        assert res_b.status_code == 403
        data_b = res_b.get_json()
        assert 'Unauthorized' in data_b.get('error', '')


def test_client_cannot_delete_another_clients_session():
    """Client B must not be allowed to delete Client A's session."""
    session_id = 'client-a-protected-session'

    with server.app.test_client() as client_a:
        res_a = client_a.post('/api/chat', json={
            'session_id': session_id,
            'message': 'Hello'
        })
        assert res_a.status_code == 200
        assert session_id in server.SESSIONS

    with server.app.test_client() as client_b:
        res_b = client_b.delete('/api/session', json={'session_id': session_id})
        assert res_b.status_code == 403
        data_b = res_b.get_json()
        assert 'Unauthorized' in data_b.get('error', '')
        # Session should still exist in server storage
        assert session_id in server.SESSIONS


def test_legitimate_client_session_continuity(monkeypatch):
    """Legitimate client must be able to continue its own session across multiple queries."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')

    with server.app.test_client() as client:
        # First query
        res1 = client.post('/api/chat', json={
            'session_id': 'legit-user-session',
            'message': 'Can I take Panadol with amoxicillin?'
        })
        assert res1.status_code == 200
        data1 = res1.get_json()
        assert 'acetaminophen' in [g.lower() for g in data1['verified_generics']]

        # Follow-up query using same client and session
        res2 = client.post('/api/chat', json={
            'session_id': 'legit-user-session',
            'message': 'What about simvastatin?'
        })
        assert res2.status_code == 200
        data2 = res2.get_json()
        assert 'simvastatin' in [g.lower() for g in data2['verified_generics']]

        # Delete own session succeeds
        res_del = client.delete('/api/session', json={'session_id': 'legit-user-session'})
        assert res_del.status_code == 200
        assert 'legit-user-session' not in server.SESSIONS


def test_new_session_generates_unpredictable_identifier(monkeypatch):
    """When client passes no session ID, server generates a secure unpredictable token."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'Hi!')

    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'Hi there'})
        assert res.status_code == 200
        data = res.get_json()
        session_id = data.get('session_id')
        assert session_id is not None
        # Must be cryptographically strong (at least 32 url-safe chars)
        assert len(session_id) >= 32
        assert session_id in server.SESSIONS


# ===========================================================================
# 2. Session Lifecycle Tests
# ===========================================================================

def test_expired_session_is_cleaned_up(monkeypatch):
    """Sessions exceeding SESSION_TTL must be pruned and not reused."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')

    with server.app.test_client() as client:
        res = client.post('/api/chat', json={
            'session_id': 'expiring-session',
            'message': 'Panadol and amoxicillin'
        })
        assert res.status_code == 200
        assert 'expiring-session' in server.SESSIONS

        # Simulate expiration by rolling back last_accessed past SESSION_TTL (3600s)
        server.SESSIONS['expiring-session']['last_accessed'] = time.time() - (server.SESSION_TTL + 60)

        # Subsequent query from another client: old session was expired and cleaned up
        with server.app.test_client() as client2:
            res2 = client2.post('/api/chat', json={
                'session_id': 'expiring-session',
                'message': 'Hello'
            })
            # It creates a fresh session for client2 since old one was pruned
            assert res2.status_code == 200
            assert server.SESSIONS['expiring-session']['owner_id'] != res.headers.get('Set-Cookie')


def test_session_message_count_limit(monkeypatch):
    """A session exceeding MAX_SESSION_MESSAGES must be rejected."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')

    with server.app.test_client() as client:
        res1 = client.post('/api/chat', json={
            'session_id': 'chatty-session',
            'message': 'Hello'
        })
        assert res1.status_code == 200

        # Simulate reaching the message limit (100)
        server.SESSIONS['chatty-session']['message_count'] = server.MAX_SESSION_MESSAGES

        res_blocked = client.post('/api/chat', json={
            'session_id': 'chatty-session',
            'message': 'Another message'
        })
        assert res_blocked.status_code == 429
        data = res_blocked.get_json()
        assert 'message limit exceeded' in data['error'].lower()


def test_session_capacity_limit(monkeypatch):
    """When MAX_TOTAL_SESSIONS is reached, new session creation is rejected."""
    monkeypatch.setattr(server, 'MAX_TOTAL_SESSIONS', 2)

    with server.app.test_client() as client:
        # Fill capacity
        res1 = client.post('/api/chat', json={'session_id': 'sess-1', 'message': 'Hi'})
        assert res1.status_code == 200
        res2 = client.post('/api/chat', json={'session_id': 'sess-2', 'message': 'Hi'})
        assert res2.status_code == 200

        # Third session exceeds capacity
        res3 = client.post('/api/chat', json={'session_id': 'sess-3', 'message': 'Hi'})
        assert res3.status_code == 503
        data = res3.get_json()
        assert 'capacity reached' in data['error'].lower()


# ===========================================================================
# 3. Rate Limiting Tests
# ===========================================================================

def test_rate_limiter_allows_normal_requests_and_blocks_flooding(monkeypatch):
    """Flooding requests past rate limit threshold must receive 429."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')
    # Temporarily set small rate limit for fast testing
    monkeypatch.setattr(server.RATE_LIMITER, 'max_requests', 5)

    with server.app.test_client() as client:
        for i in range(5):
            res = client.post('/api/chat', json={'message': f'Message {i}'})
            assert res.status_code == 200, f'Request {i} failed unexpectedly'

        # 6th request should exceed the limit
        res_blocked = client.post('/api/chat', json={'message': 'Flood message'})
        assert res_blocked.status_code == 429
        assert 'Rate limit exceeded' in res_blocked.get_json()['error']
        assert 'Retry-After' in res_blocked.headers


# ===========================================================================
# 4. Message & Payload Size Limits Tests
# ===========================================================================

def test_normal_message_succeeds(monkeypatch):
    """Normal medication query succeeds without rejection."""
    monkeypatch.setattr(server, 'generate_reply', lambda msg, res, hist: 'ok')

    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'Can I take Panadol with amoxicillin?'})
        assert res.status_code == 200


def test_oversized_message_rejected_without_truncation():
    """Message exceeding MAX_MESSAGE_LENGTH (4096 chars) is rejected with 413."""
    oversized_message = 'A' * (server.MAX_MESSAGE_LENGTH + 100)

    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': oversized_message})
        assert res.status_code == 413
        data = res.get_json()
        assert 'exceeds the maximum allowed limit' in data['error']


def test_oversized_json_payload_rejected():
    """JSON payload exceeding MAX_PAYLOAD_BYTES (64KB) is rejected with 413."""
    large_payload = {
        'message': 'Hello',
        'padding': 'X' * (server.MAX_PAYLOAD_BYTES + 1024)
    }

    with server.app.test_client() as client:
        res = client.post('/api/chat', json=large_payload)
        assert res.status_code == 413
        data = res.get_json()
        assert 'payload exceeds maximum allowed size' in data['error'].lower()


# ===========================================================================
# 5. Deployment Readiness Tests (CORS & Cookie Security)
# ===========================================================================

def test_cors_localhost_allowed_and_disallowed_blocked():
    """Verify that localhost origin is permitted and unknown origins are not."""
    with server.app.test_client() as client:
        res = client.get('/health', headers={'Origin': 'http://localhost:5000'})
        # Note: /api/* is governed by CORS rules
        res_api = client.post('/api/chat', json={'message': 'Hi'}, headers={'Origin': 'http://localhost:5000'})
        assert res_api.headers.get('Access-Control-Allow-Origin') == 'http://localhost:5000'

        res_evil = client.post('/api/chat', json={'message': 'Hi'}, headers={'Origin': 'http://evil.com'})
        assert res_evil.headers.get('Access-Control-Allow-Origin') is None


def test_allowed_origins_parsing_filters_wildcard(monkeypatch):
    """Verify ALLOWED_ORIGINS env parsing strips whitespace and blocks wildcard '*'."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://dosewise.pythonanywhere.com, http://example.com, *")
    origins = server._get_allowed_origins()
    assert "https://dosewise.pythonanywhere.com" in origins
    assert "http://example.com" in origins
    assert "*" not in origins


def test_cookie_secure_flag_respects_deployment_mode(monkeypatch):
    """Verify cookie gets Secure attribute when SECURE_COOKIES or production env is set."""
    monkeypatch.setenv("SECURE_COOKIES", "true")
    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'Hi'})
        cookie_header = res.headers.get('Set-Cookie', '')
        assert 'Secure' in cookie_header
        assert 'HttpOnly' in cookie_header
        assert 'SameSite=Lax' in cookie_header


def test_cookie_secure_flag_defaults_to_false_for_local_dev(monkeypatch):
    """Verify cookie omits Secure attribute by default for local development & testing."""
    monkeypatch.delenv("SECURE_COOKIES", raising=False)
    monkeypatch.delenv("DOSEWISE_ENV", raising=False)
    with server.app.test_client() as client:
        res = client.post('/api/chat', json={'message': 'Hi'})
        cookie_header = res.headers.get('Set-Cookie', '')
        assert 'Secure' not in cookie_header
        assert 'HttpOnly' in cookie_header
        assert 'SameSite=Lax' in cookie_header


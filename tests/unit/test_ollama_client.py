"""
tests/unit/test_ollama_client.py
==================================
Unit tests for infra/ollama_client.py (all tests use mock — no real Ollama required)

Tests:
  1.  temperature != 0.0 raises ValueError immediately (no network call)
  2.  chat() with temperature=0.0 succeeds (mocked response)
  3.  generate() with temperature=0.0 succeeds (mocked response)
  4.  OllamaResponse.parse_json() parses valid JSON
  5.  OllamaResponse.parse_json() raises json.JSONDecodeError on invalid JSON
  6.  content_hash is SHA-256 of response text
  7.  processing_ms is a non-negative integer
  8.  Semaphore is acquired once per call (thread-safety)
  9.  HTTP 500 from Ollama raises OllamaError
  10. Timeout raises requests.Timeout
  11. prompt_version embedded in request system message
  12. health_check returns False when Ollama unreachable (mocked)
"""
import hashlib
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from infra.ollama_client import OllamaClient, OllamaError, _OLLAMA_SEMAPHORE


def _make_chat_response(content: str = '{"status": "VALID"}') -> MagicMock:
    """Build a mock requests.Response for /api/chat."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "model": "llama3.1:8b-instruct-q4_K_M",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 100,
        "eval_count": 50,
    }
    return mock_resp


def _make_generate_response(text: str = "Narrative text.") -> MagicMock:
    """Build a mock requests.Response for /api/generate."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "model": "llama3.1:8b-instruct-q4_K_M",
        "response": text,
        "done": True,
        "prompt_eval_count": 80,
        "eval_count": 30,
    }
    return mock_resp


@pytest.fixture()
def client():
    return OllamaClient(host="http://localhost:11434")


# ─────────────────────────────────────────────────────────────────────────────
# 1. temperature != 0.0 raises ValueError immediately
# ─────────────────────────────────────────────────────────────────────────────

def test_nonzero_temperature_raises(client):
    with pytest.raises(ValueError, match="temperature=0.7 rejected"):
        client.chat(
            system_prompt="You are a PV expert.",
            user_message="Classify this.",
            temperature=0.7,
        )


def test_generate_nonzero_temperature_raises(client):
    with pytest.raises(ValueError, match="temperature=0.5 rejected"):
        client.generate(prompt="Generate narrative.", temperature=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. chat() succeeds with mocked response
# ─────────────────────────────────────────────────────────────────────────────

def test_chat_succeeds_with_mock(client):
    content = '{"status": "VALID", "tier": "TIER_1"}'
    with patch.object(client._session, "post", return_value=_make_chat_response(content)) as mock_post:
        response = client.chat(
            system_prompt="You are a PV triage expert.",
            user_message="Is this a valid case?",
            prompt_version="triage-prompt-v1.0",
        )
    assert response.text == content
    assert mock_post.called


# ─────────────────────────────────────────────────────────────────────────────
# 3. generate() succeeds with mocked response
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_succeeds_with_mock(client):
    text = "The patient experienced anaphylaxis after amoxicillin."
    with patch.object(client._session, "post", return_value=_make_generate_response(text)):
        response = client.generate(
            prompt="Generate an E2B narrative.",
            prompt_version="narrative-prompt-v1.0",
        )
    assert response.text == text


# ─────────────────────────────────────────────────────────────────────────────
# 4. parse_json() on valid JSON
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_json_valid(client):
    content = '{"risk_tier": "TIER_1", "confidence": 0.97}'
    with patch.object(client._session, "post", return_value=_make_chat_response(content)):
        response = client.chat("sys", "user", "v1.0")
    data = response.parse_json()
    assert data["risk_tier"] == "TIER_1"
    assert data["confidence"] == pytest.approx(0.97)


# ─────────────────────────────────────────────────────────────────────────────
# 5. parse_json() raises on invalid JSON
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_json_invalid_raises(client):
    content = "this is not JSON { badly formatted"
    with patch.object(client._session, "post", return_value=_make_chat_response(content)):
        response = client.chat("sys", "user", "v1.0")
    with pytest.raises(json.JSONDecodeError):
        response.parse_json()


# ─────────────────────────────────────────────────────────────────────────────
# 6. content_hash is SHA-256 of response text
# ─────────────────────────────────────────────────────────────────────────────

def test_content_hash_is_sha256(client):
    content = '{"status": "VALID"}'
    expected_hash = hashlib.sha256(content.encode()).hexdigest()
    with patch.object(client._session, "post", return_value=_make_chat_response(content)):
        response = client.chat("sys", "user", "v1.0")
    assert response.content_hash == expected_hash
    assert len(response.content_hash) == 64


# ─────────────────────────────────────────────────────────────────────────────
# 7. processing_ms is non-negative integer
# ─────────────────────────────────────────────────────────────────────────────

def test_processing_ms_non_negative(client):
    with patch.object(client._session, "post", return_value=_make_chat_response()):
        response = client.chat("sys", "user", "v1.0")
    assert isinstance(response.processing_ms, int)
    assert response.processing_ms >= 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Global semaphore serializes concurrent calls
# ─────────────────────────────────────────────────────────────────────────────

def test_semaphore_serializes_concurrent_calls(client):
    """
    Verify the global Semaphore(1) prevents concurrent Ollama calls.
    Patch requests.Session.post at the class level so both threads share the mock.
    """
    import threading as _t
    counter        = [0]
    max_concurrent = [0]
    lock           = _t.Lock()

    def slow_post(self_session, url, json=None, **kwargs):
        with lock:
            counter[0] += 1
            if counter[0] > max_concurrent[0]:
                max_concurrent[0] = counter[0]
        time.sleep(0.05)
        with lock:
            counter[0] -= 1
        return _make_chat_response()

    results = []
    errors  = []

    def thread_fn():
        try:
            r = client.chat("sys", "user", "v1.0")
            results.append(r)
        except Exception as exc:
            errors.append(exc)

    import requests as _req
    with patch.object(_req.Session, "post", slow_post):
        t1 = _t.Thread(target=thread_fn)
        t2 = _t.Thread(target=thread_fn)
        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    assert len(results) == 2
    assert max_concurrent[0] <= 1, (
        f"Semaphore failed: max_concurrent={max_concurrent[0]}, expected ≤ 1"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. HTTP 500 raises OllamaError
# ─────────────────────────────────────────────────────────────────────────────

def test_http_500_raises_ollama_error(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

    with patch.object(client._session, "post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="HTTP 500"):
            client.chat("sys", "user", "v1.0")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Timeout raises requests.Timeout
# ─────────────────────────────────────────────────────────────────────────────

def test_timeout_raises(client):
    with patch.object(
        client._session, "post",
        side_effect=requests.Timeout("Connection timed out")
    ):
        with pytest.raises(requests.Timeout):
            client.chat("sys", "user", "v1.0")


# ─────────────────────────────────────────────────────────────────────────────
# 11. prompt_version embedded in system message
# ─────────────────────────────────────────────────────────────────────────────

def test_prompt_version_embedded_in_system_message(client):
    captured_payload = {}

    def fake_post(url, json=None, **kwargs):
        captured_payload.update(json or {})
        return _make_chat_response()

    with patch.object(client._session, "post", side_effect=fake_post):
        client.chat(
            system_prompt="You are an expert.",
            user_message="Classify this.",
            prompt_version="triage-prompt-v2.0",
        )

    system_msg = captured_payload["messages"][0]["content"]
    assert "triage-prompt-v2.0" in system_msg
    assert "You are an expert." in system_msg


# ─────────────────────────────────────────────────────────────────────────────
# 12. health_check returns False when Ollama unreachable
# ─────────────────────────────────────────────────────────────────────────────

def test_health_check_false_when_unreachable(client):
    with patch.object(
        client._session, "get",
        side_effect=requests.ConnectionError("Connection refused")
    ):
        assert client.health_check() is False

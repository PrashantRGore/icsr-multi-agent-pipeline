"""
infra/ollama_client.py
======================
Serialized Ollama REST API client.

CRITICAL HARDWARE RULE (13th Gen i5 / 16GB RAM):
  Only ONE Ollama inference call may be active at any time.
  A threading.Semaphore(1) enforces this as a hard system-level guard.
  Violating this causes OOM — the Llama 3.1 8B Q4_K_M model alone uses ~5.5GB.

Design decisions (v4):
  - temperature=0.0 hardcoded — deterministic outputs required for PV.
    Any attempt to call with temperature != 0.0 raises ValueError immediately.
  - model_name defaults to OLLAMA_MODEL env var (llama3.1:8b-instruct-q4_K_M).
  - Structured output (JSON) via Ollama's /api/chat with format="json".
  - prompt_version is passed as a comment in the system prompt for auditability;
    it does NOT affect temperature or output — it is purely a governance marker.
  - Timeout: configurable via OLLAMA_TIMEOUT_SECONDS env var (default 120s).
    LangGraph does not retry by default; agents handle timeouts by routing to HITL.
  - SHA-256 content_hash is computed on the response text for AuditLogEntry.
  - processing_ms is measured with time.perf_counter() for AuditLogEntry.

Ollama API reference:
  POST /api/chat   — chat completion (used for structured output)
  POST /api/generate — raw text generation (used for narrative agent)
  GET  /api/tags   — list available models (used for health check)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global serializer — ONE active Ollama call at a time
# ─────────────────────────────────────────────────────────────────────────────

_OLLAMA_SEMAPHORE = threading.Semaphore(1)
"""
Hardware guard: 16GB RAM, Llama 3.1 8B Q4_K_M = ~5.5GB.
Two simultaneous inference calls would saturate RAM and cause OOM.
This semaphore is MODULE-LEVEL — shared across all OllamaClient instances.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
_DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))


# ─────────────────────────────────────────────────────────────────────────────
# Response dataclass
# ─────────────────────────────────────────────────────────────────────────────

class OllamaResponse:
    """Structured result from an Ollama inference call."""

    __slots__ = ("text", "content_hash", "processing_ms", "model", "prompt_eval_count",
                 "eval_count", "raw")

    def __init__(
        self,
        text: str,
        content_hash: str,
        processing_ms: int,
        model: str,
        prompt_eval_count: Optional[int],
        eval_count: Optional[int],
        raw: dict,
    ) -> None:
        self.text              = text
        self.content_hash      = content_hash   # SHA-256 of text, for AuditLogEntry
        self.processing_ms     = processing_ms
        self.model             = model
        self.prompt_eval_count = prompt_eval_count
        self.eval_count        = eval_count
        self.raw               = raw             # Full Ollama API response dict

    def parse_json(self) -> Any:
        """Parse self.text as JSON. Raises json.JSONDecodeError if malformed."""
        return json.loads(self.text)

    def __repr__(self) -> str:
        return (
            f"OllamaResponse(model={self.model!r}, "
            f"processing_ms={self.processing_ms}, "
            f"tokens={self.eval_count}, "
            f"hash={self.content_hash[:8]}...)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# OllamaClient
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Serialized Ollama client.

    Usage:
        client = OllamaClient()
        response = client.chat(
            system_prompt="You are a PV expert...",
            user_message="Classify this narrative...",
            prompt_version="triage-prompt-v1.0",
        )
        data = response.parse_json()  # Structured output

    Raises:
        ValueError:      temperature != 0.0 attempted
        OllamaError:     Ollama returned non-200 or malformed response
        requests.Timeout: call exceeded OLLAMA_TIMEOUT_SECONDS
    """

    def __init__(
        self,
        host:       str = _DEFAULT_HOST,
        model:      str = _DEFAULT_MODEL,
        timeout_s:  int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.host      = host.rstrip("/")
        self.model     = model
        self.timeout_s = timeout_s
        self._session  = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ── Health ───────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Returns True if Ollama is reachable and the configured model is available.
        Does NOT acquire the semaphore (read-only query).
        """
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            models = {m["name"] for m in resp.json().get("models", [])}
            if self.model not in models:
                logger.warning(
                    "OllamaClient: model %r not in available models: %s",
                    self.model, sorted(models)
                )
                return False
            return True
        except Exception as exc:
            logger.warning("OllamaClient health_check failed: %s", exc)
            return False

    # ── Chat (structured JSON output) ────────────────────────────────────────

    def chat(
        self,
        system_prompt:  str,
        user_message:   str,
        prompt_version: str = "unknown",
        temperature:    float = 0.0,
    ) -> OllamaResponse:
        """
        Send a chat completion request to Ollama (/api/chat) with format="json".

        DETERMINISM CONTRACT: temperature MUST be 0.0. Any other value raises
        ValueError immediately, before the semaphore is acquired.

        prompt_version is embedded as a comment in the system prompt for
        21 CFR Part 11 / CIOMS WG XIV prompt traceability. It does not
        affect the model's output.
        """
        if temperature != 0.0:
            raise ValueError(
                f"OllamaClient: temperature={temperature} rejected. "
                "All PV inference calls must use temperature=0.0 for determinism. "
                "This is a hard compliance requirement."
            )

        # Embed prompt_version as a governance marker in the system prompt
        versioned_system = (
            f"[prompt_version: {prompt_version}]\n\n"
            f"{system_prompt}"
        )

        payload = {
            "model":  self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 4096},
            "messages": [
                {"role": "system",  "content": versioned_system},
                {"role": "user",    "content": user_message},
            ],
        }

        return self._call("/api/chat", payload, extract_text=lambda r: r["message"]["content"])

    # ── Generate (raw text — for NarrativeAgent) ─────────────────────────────

    def generate(
        self,
        prompt:         str,
        prompt_version: str = "unknown",
        temperature:    float = 0.0,
    ) -> OllamaResponse:
        """
        Send a raw generation request to Ollama (/api/generate).
        Used by the NarrativeAgent to produce free-form E2B narrative text.
        """
        if temperature != 0.0:
            raise ValueError(
                f"OllamaClient: temperature={temperature} rejected. "
                "temperature=0.0 is mandatory for all PV inference."
            )

        versioned_prompt = f"[prompt_version: {prompt_version}]\n\n{prompt}"

        payload = {
            "model":   self.model,
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 2048},
            "prompt":  versioned_prompt,
        }

        return self._call("/api/generate", payload, extract_text=lambda r: r["response"])

    # ── Internal ─────────────────────────────────────────────────────────────

    def _call(
        self,
        endpoint:     str,
        payload:      dict,
        extract_text: Any,
    ) -> OllamaResponse:
        """
        Acquire the global semaphore, make the HTTP call, release.
        Semaphore ensures no concurrent Ollama calls (hardware constraint).
        """
        logger.debug(
            "OllamaClient: acquiring semaphore for %s on %s",
            endpoint, self.model
        )
        with _OLLAMA_SEMAPHORE:
            t0 = time.perf_counter()
            try:
                resp = self._session.post(
                    f"{self.host}{endpoint}",
                    json=payload,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
            except requests.Timeout:
                logger.error(
                    "OllamaClient: timeout after %ds on %s", self.timeout_s, endpoint
                )
                raise
            except requests.HTTPError as exc:
                logger.error("OllamaClient: HTTP %s on %s: %s", resp.status_code, endpoint, exc)
                raise OllamaError(f"HTTP {resp.status_code}: {resp.text[:300]}") from exc

            processing_ms = int((time.perf_counter() - t0) * 1000)
            raw = resp.json()

        # Extract text and compute hash outside the semaphore (CPU only)
        try:
            text = extract_text(raw)
        except (KeyError, TypeError) as exc:
            raise OllamaError(
                f"OllamaClient: unexpected response structure from {endpoint}: {exc}"
            ) from exc

        content_hash = hashlib.sha256(text.encode()).hexdigest()

        logger.debug(
            "OllamaClient: %s completed in %dms (hash=%s...)",
            endpoint, processing_ms, content_hash[:8]
        )

        return OllamaResponse(
            text=text,
            content_hash=content_hash,
            processing_ms=processing_ms,
            model=raw.get("model", self.model),
            prompt_eval_count=raw.get("prompt_eval_count"),
            eval_count=raw.get("eval_count"),
            raw=raw,
        )


class OllamaError(Exception):
    """Raised when Ollama returns an unexpected response or non-2xx status."""

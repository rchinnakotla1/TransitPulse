"""Phase 4: OpenAI client with key-health checking, error classification and logging.

Design goals
* The dashboard must **never crash** because of the LLM: every call returns a result or a classified error.
* The user sees *why* the LLM is unavailable (expired / invalid key, quota, rate limit, network, unknown model)
  through ``health_check()``, which the Streamlit sidebar shows as a status badge.
* Every call and every failure is written to ``logs/llm.log`` (never the key itself).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass

from src.config import LOG_DIR, ROOT

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

DEFAULT_MODEL = "gpt-4o-mini"

log = logging.getLogger("transitpulse.llm")
if not log.handlers:
    h = logging.FileHandler(LOG_DIR / "llm.log", encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


@dataclass
class LLMStatus:
    ok: bool
    code: str            # ok | missing_key | invalid_or_expired_key | quota_exceeded | rate_limited | network | model_not_found | error
    message: str         # human-readable, shown in the app
    model: str
    checked_at: float

    def to_dict(self):
        return asdict(self)


MESSAGES = {
    "ok": "OpenAI connected.",
    "missing_key": "No OpenAI API key found (set OPENAI_API_KEY in .env). Using template explanations.",
    "invalid_or_expired_key": "OpenAI API key is invalid, revoked or expired (HTTP 401). Using template explanations. "
                              "Create a new key at platform.openai.com and update .env.",
    "quota_exceeded": "OpenAI quota / billing limit reached (HTTP 429 insufficient_quota). Using template explanations.",
    "rate_limited": "OpenAI rate limit hit (HTTP 429). Retrying later; using template explanations for now.",
    "network": "Cannot reach the OpenAI API (network / timeout). Using template explanations.",
    "model_not_found": "The configured OpenAI model is not available for this key (HTTP 404). Set OPENAI_MODEL in .env.",
    "error": "Unexpected OpenAI error. Using template explanations.",
}


def api_key() -> str | None:
    k = os.getenv("OPENAI_API_KEY", "").strip()
    return k or None


def model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def classify_error(exc: Exception) -> str:
    """Map an exception from the openai SDK (or a generic one) to a status code. Works without importing openai."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    body = str(getattr(exc, "body", "") or "") + " " + str(exc)
    if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
        return "invalid_or_expired_key"
    if name == "NotFoundError" or status == 404:
        return "model_not_found"
    if name == "RateLimitError" or status == 429:
        return "quota_exceeded" if "insufficient_quota" in body or "quota" in body.lower() else "rate_limited"
    if name in ("APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError", "ConnectError"):
        return "network"
    return "error"


def _client():
    from openai import OpenAI
    return OpenAI(api_key=api_key(), timeout=20, max_retries=1)


_STATUS: LLMStatus | None = None


def health_check(force: bool = False, ttl: int = 600) -> LLMStatus:
    """Cheap check (models.retrieve) cached for ``ttl`` seconds."""
    global _STATUS
    if _STATUS and not force and time.time() - _STATUS.checked_at < ttl:
        return _STATUS
    m = model_name()
    if not api_key():
        _STATUS = LLMStatus(False, "missing_key", MESSAGES["missing_key"], m, time.time())
    else:
        try:
            _client().models.retrieve(m)
            _STATUS = LLMStatus(True, "ok", MESSAGES["ok"], m, time.time())
        except Exception as e:  # noqa: BLE001 - we classify everything
            code = classify_error(e)
            _STATUS = LLMStatus(False, code, MESSAGES[code], m, time.time())
            log.warning("health_check failed: %s (%s: %s)", code, type(e).__name__, str(e)[:200])
    if _STATUS.ok:
        log.info("health_check ok model=%s", m)
    return _STATUS


def mark_failed(code: str) -> None:
    """Called when a real request fails, so the badge updates without waiting for the TTL."""
    global _STATUS
    _STATUS = LLMStatus(False, code, MESSAGES.get(code, MESSAGES["error"]), model_name(), time.time())


def chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 350) -> tuple[str | None, str]:
    """Returns (text, status_code). text is None on failure; never raises."""
    if not api_key():
        return None, "missing_key"
    t = time.time()
    try:
        r = _client().chat.completions.create(
            model=model_name(), temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        text = r.choices[0].message.content.strip()
        usage = getattr(r, "usage", None)
        log.info("chat ok model=%s %.1fs tokens=%s", model_name(), time.time() - t,
                 getattr(usage, "total_tokens", "?"))
        return text, "ok"
    except Exception as e:  # noqa: BLE001
        code = classify_error(e)
        mark_failed(code)
        log.warning("chat failed: %s (%s: %s)", code, type(e).__name__, str(e)[:200])
        return None, code


if __name__ == "__main__":
    s = health_check(force=True)
    print(s.code, "-", s.message, f"(model {s.model})")

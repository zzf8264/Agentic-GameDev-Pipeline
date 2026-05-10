"""
LLM Client — OpenAI-compatible wrapper with retry, token tracking & dry-run mode.

Wraps ``openai.AsyncOpenAI`` with:
  - Exponential backoff retry via ``tenacity``.
  - Token estimation via ``tiktoken`` (falls back to 4-char heuristic).
  - Configurable model, endpoint, timeout.
  - ``dry_run=True`` mode that returns None instead of calling the API,
    allowing agents to exercise their offline fallback paths.

Usage:
    llm = LLMClient(model="gpt-4o", endpoint="http://localhost:11434/v1")
    reply = await llm.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("core.llm")

# ---------------------------------------------------------------------------
# Lazy third-party imports — modules are only imported when the client is
# actually instantiated, so importing core.llm never fails at module level
# even when openai / tenacity are not installed.
# ---------------------------------------------------------------------------

_AsyncOpenAI = None
_tenacity_retry = None


def _ensure_deps():
    global _AsyncOpenAI, _tenacity_retry
    if _AsyncOpenAI is not None:
        return
    try:
        from openai import AsyncOpenAI as _AO

        _AsyncOpenAI = _AO
    except ImportError as e:
        raise ImportError("openai is required: pip install openai") from e

    try:
        from tenacity import (
            retry as _retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_exception_type,
        )

        _tenacity_retry = _retry
        _tenacity_retry.kwargs = {  # type: ignore[attr-defined]
            "stop": stop_after_attempt(3),
            "wait": wait_exponential(multiplier=1, min=1, max=10),
            "retry": retry_if_exception_type((TimeoutError, ConnectionError)),
            "before_sleep": lambda rs: logger.warning(
                "LLM call failed (attempt %d): %s",
                rs.attempt_number,
                rs.outcome.exception() if rs.outcome else "unknown",
            ),
        }
    except ImportError as e:
        raise ImportError("tenacity is required: pip install tenacity") from e


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_HAS_TIKTOKEN = False
try:
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:
    pass


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Token count via tiktoken, falling back to 4-char heuristic."""
    if _HAS_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
    return len(text) // 4


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper around ``AsyncOpenAI`` with retry, metrics, and dry-run."""

    def __init__(
        self,
        model: str = "gpt-4o",
        endpoint: Optional[str] = None,
        api_key: str = "sk-placeholder",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout_s: float = 120.0,
        dry_run: bool = False,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dry_run = dry_run

        if not dry_run:
            _ensure_deps()
            base_url = endpoint or os.getenv("LLM_ENDPOINT", "http://localhost:11434/v1")
            self._client = _AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        else:
            self._client = None  # type: ignore

        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._call_count: int = 0

        logger.info(
            "LLMClient [model=%s, dry_run=%s]",
            model,
            dry_run,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Send a chat-completion request. Returns content or None on dry-run / failure."""
        if self.dry_run:
            tok = sum(estimate_tokens(m.get("content", ""), self.model) for m in messages)
            self._total_prompt_tokens += tok
            self._call_count += 1
            logger.info("DRY_RUN — skip LLM call (est. %d prompt tokens)", tok)
            return None

        full_messages = list(messages)
        if system_prompt:
            full_messages.insert(0, {"role": "system", "content": system_prompt})

        try:

            @_tenacity_retry(**_tenacity_retry.kwargs)  # type: ignore
            async def _do_call():
                return await self._client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                )

            resp = await _do_call()
            content = resp.choices[0].message.content
            usage = resp.usage

            if usage:
                self._total_prompt_tokens += usage.prompt_tokens
                self._total_completion_tokens += usage.completion_tokens
            else:
                self._total_prompt_tokens += sum(
                    estimate_tokens(m.get("content", ""), self.model) for m in full_messages
                )
                self._total_completion_tokens += estimate_tokens(content or "", self.model)

            self._call_count += 1
            logger.info(
                "LLM OK [call=%d] ptok=%s ctok=%s",
                self._call_count,
                usage.prompt_tokens if usage else "?",
                usage.completion_tokens if usage else "?",
            )
            return content

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            self._call_count += 1
            return None

    async def chat_simple(self, user_prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Convenience: single user message, returns reply."""
        return await self.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
        )

    # ── Metrics ────────────────────────────────────────────────────────────

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens

    @property
    def call_count(self) -> int:
        return self._call_count

    def snapshot(self) -> dict:
        return {
            "model": self.model,
            "call_count": self.call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    async def aclose(self) -> None:
        if self._client:
            await self._client.close()

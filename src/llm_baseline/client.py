"""Mockable OpenAI-compatible Qwen client; no transport is created implicitly."""

from __future__ import annotations

import os
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from typing import Any, Protocol

from .prompt import ParsedDecision, parse_response


QWEN_MODEL = "qwen3.8-flash"
QWEN_BASE_URL = "https://ws-b72yc4qdpbcg0dpv.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


class Transport(Protocol):
    def post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]: ...


class UrllibTransport:
    """Standard-library transport for later AutoDL use; never instantiated by tests."""

    def post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=90) as response:  # nosec B310: configured HTTPS endpoint
            value = json.loads(response.read().decode("utf-8"))
            request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
        if not isinstance(value, dict):
            raise ValueError("OpenAI-compatible response is not a JSON object")
        if request_id:
            value["_qwen_provider_request_id"] = request_id
        return value


@dataclass(frozen=True)
class QwenConfig:
    model: str = QWEN_MODEL
    base_url: str = QWEN_BASE_URL
    temperature: float = 0.0
    enable_thinking: bool = False
    stream: bool = False
    request_json_object: bool = True
    max_retries: int = 3

    def public_dict(self) -> dict[str, Any]:
        return {
            "model": self.model, "base_url": self.base_url, "temperature": self.temperature,
            "enable_thinking": self.enable_thinking, "stream": self.stream,
            "request_json_object": self.request_json_object, "max_retries": self.max_retries,
        }


class QwenClient:
    """A client requiring an explicit injected transport and environment key."""

    def __init__(self, config: QwenConfig, transport: Transport, api_key: str | None = None) -> None:
        self.config = config
        self.transport = transport
        self._api_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY")

    def request_payload(self, prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "stream": self.config.stream,
            "enable_thinking": self.config.enable_thinking,
        }
        if self.config.request_json_object:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def classify(self, prompt: str) -> tuple[ParsedDecision, dict[str, Any] | None]:
        if not self._api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required only when executing Qwen requests")
        payload = self.request_payload(prompt)
        response = self.transport.post_json(
            self.config.base_url.rstrip("/") + "/chat/completions",
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}, payload,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            return ParsedDecision("malformed", None, f"missing chat-completion content: {exc}"), response
        if not isinstance(content, str):
            return ParsedDecision("malformed", None, "chat-completion content is not text"), response
        return parse_response(content), response

    def _safe_error(self, exc: Exception) -> str:
        message = str(exc)
        if self._api_key:
            message = message.replace(self._api_key, "[REDACTED]")
        return f"{type(exc).__name__}: {message}"

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code == 429 or 500 <= exc.code <= 599
        return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        """Return a provider Retry-After delay when a retryable HTTP error supplies one."""
        if not isinstance(exc, urllib.error.HTTPError):
            return None
        value = exc.headers.get("Retry-After") if exc.headers is not None else None
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                target = parsedate_to_datetime(value)
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, IndexError, OverflowError):
                return None

    def classify_with_retries(
        self, prompt: str, sleep: Any = time.sleep, before_attempt: Any | None = None,
    ) -> tuple[ParsedDecision, dict[str, Any] | None, int, str | None]:
        """Retry transport errors and malformed output a bounded number of times.

        A valid ``uncertain`` decision is never retried. A malformed response
        gets the same bounded retry policy as a transient transport failure.
        """
        last_error = None
        for attempt in range(1, self.config.max_retries + 2):
            try:
                if before_attempt is not None:
                    before_attempt()
                parsed, response = self.classify(prompt)
                if parsed.status == "malformed" and attempt <= self.config.max_retries:
                    sleep(min(2 ** (attempt - 1), 8))
                    continue
                return parsed, response, attempt - 1, None
            except Exception as exc:  # transport-specific exception types vary by provider
                if isinstance(exc, RuntimeError) and "DASHSCOPE_API_KEY" in str(exc):
                    raise
                last_error = self._safe_error(exc)
                if attempt <= self.config.max_retries and self._retryable(exc):
                    retry_after = self._retry_after_seconds(exc)
                    sleep(max(min(2 ** (attempt - 1), 8), retry_after or 0.0))
                    continue
                return ParsedDecision("transport_error", None, last_error), None, attempt - 1, last_error
        return ParsedDecision("transport_error", None, last_error), None, self.config.max_retries, last_error

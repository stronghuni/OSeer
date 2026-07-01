"""OpenAI-compatible chat-completions provider (default: FriendliAI)."""

from __future__ import annotations

import asyncio

import httpx

from ..config import Settings, get_settings
from .base import Message, ModelProvider, ProviderError

# Retried on these transient statuses.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class FriendliProvider(ModelProvider):
    """Calls ``{api_base}/chat/completions`` with a bearer token.

    Works against FriendliAI serverless or a dedicated endpoint, or any other
    OpenAI-compatible host, purely via :class:`~oseer.config.Settings`.
    """

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._settings = settings or get_settings()
        self._client = client  # injectable for tests

    def _build_payload(self, messages: list[Message], reasoning: bool) -> dict:
        s = self._settings
        payload: dict = {
            "model": s.model,
            "messages": messages,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "max_tokens": s.reasoning_max_tokens if reasoning else s.max_tokens,
            "stream": False,
        }
        # top_k is a common vendor extension; vLLM/SGLang accept it.
        if s.send_top_k and s.top_k:
            payload["top_k"] = s.top_k
        # Control the model's <think> mode per request (Qwen chat-template convention).
        if s.send_thinking_flag:
            payload["chat_template_kwargs"] = {"enable_thinking": reasoning}
        # Force syntactically valid JSON when we are NOT reasoning (think traces need free text).
        if s.json_mode and not reasoning:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def complete(self, messages: list[Message], reasoning: bool = False) -> str:
        s = self._settings

        url = s.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        # Auth is optional: self-hosted servers typically need no key.
        if s.api_key_present():
            headers["Authorization"] = f"Bearer {s.api_key}"
        payload = self._build_payload(messages, reasoning)

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=s.timeout)
        try:
            return await self._request_with_retries(client, url, headers, payload)
        finally:
            if owns_client:
                await client.aclose()

    async def _request_with_retries(
        self, client: httpx.AsyncClient, url: str, headers: dict, payload: dict
    ) -> str:
        s = self._settings
        last_exc: Exception | None = None
        for attempt in range(s.retries + 1):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in _RETRY_STATUS and attempt < s.retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                resp.raise_for_status()
                return self._extract(resp.json())
            except httpx.HTTPStatusError as exc:
                # Non-retryable status (4xx other than throttling) → fail fast.
                raise ProviderError(
                    f"Model API returned {exc.response.status_code}: {exc.response.text[:300]}"
                ) from exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < s.retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
        raise ProviderError(f"Model API request failed: {last_exc}") from last_exc

    @staticmethod
    def _extract(body: dict) -> str:
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected API response shape: {body!r:.300}") from exc

        content = (message.get("content") or "").strip()
        if content:
            return content
        # Some servers with a reasoning parser return the think trace separately and leave
        # content empty; the JSON we want may be embedded in reasoning_content as a fallback.
        reasoning = (message.get("reasoning_content") or "").strip()
        if reasoning:
            return reasoning
        raise ProviderError("Model returned an empty completion.")

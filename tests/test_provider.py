"""Tests for the OpenAI-compatible provider using mocked HTTP (respx)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from oseer.config import Settings
from oseer.providers.base import ProviderError
from oseer.providers.friendli import FriendliProvider

BASE = "https://api.test/v1"


def _settings(**kw) -> Settings:
    base = dict(api_base=BASE, api_key="test-key", model="test-model", retries=2, timeout=5)
    base.update(kw)
    return Settings(**base)


def _ok(content: str, reasoning_content: str | None = None) -> httpx.Response:
    message = {"content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return httpx.Response(200, json={"choices": [{"message": message}]})


@respx.mock
async def test_complete_sends_correct_request_and_returns_content():
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=_ok("hello"))
    provider = FriendliProvider(_settings())
    out = await provider.complete([{"role": "user", "content": "hi"}])

    assert out == "hello"
    assert route.called
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer test-key"
    body = json.loads(req.content)
    assert body["model"] == "test-model"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.6
    assert body["top_k"] == 20
    assert body["stream"] is False


@respx.mock
async def test_keyless_self_hosted_omits_auth_header():
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=_ok("ok"))
    provider = FriendliProvider(_settings(api_key=""))
    out = await provider.complete([{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert "authorization" not in route.calls.last.request.headers


@respx.mock
async def test_json_mode_on_when_not_reasoning():
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=_ok("{}"))
    provider = FriendliProvider(_settings())
    await provider.complete([{"role": "user", "content": "hi"}], reasoning=False)
    body = json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 2048


@respx.mock
async def test_reasoning_enables_think_and_drops_json_mode_and_bumps_tokens():
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=_ok("<think>x</think>{}"))
    provider = FriendliProvider(_settings(reasoning_max_tokens=8192))
    await provider.complete([{"role": "user", "content": "hi"}], reasoning=True)
    body = json.loads(route.calls.last.request.content)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "response_format" not in body        # can't force JSON while thinking
    assert body["max_tokens"] == 8192           # bumped for the think trace


@respx.mock
async def test_compat_toggles_off_strips_extensions():
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=_ok("{}"))
    provider = FriendliProvider(
        _settings(send_top_k=False, send_thinking_flag=False, json_mode=False)
    )
    await provider.complete([{"role": "user", "content": "hi"}])
    body = json.loads(route.calls.last.request.content)
    for stripped in ("top_k", "chat_template_kwargs", "response_format"):
        assert stripped not in body


@respx.mock
async def test_falls_back_to_reasoning_content_when_content_empty():
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=_ok("", reasoning_content='{"exit_code": 0}')
    )
    provider = FriendliProvider(_settings())
    out = await provider.complete([{"role": "user", "content": "hi"}])
    assert out == '{"exit_code": 0}'
    assert route.called


@respx.mock
async def test_empty_completion_raises():
    respx.post(f"{BASE}/chat/completions").mock(return_value=_ok(""))
    provider = FriendliProvider(_settings())
    with pytest.raises(ProviderError, match="empty completion"):
        await provider.complete([{"role": "user", "content": "hi"}])


@respx.mock
async def test_retries_on_429_then_succeeds():
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[httpx.Response(429), httpx.Response(429), _ok("recovered")]
    )
    async with httpx.AsyncClient() as client:
        provider = FriendliProvider(_settings(retries=2), client=client)
        out = await provider.complete([{"role": "user", "content": "hi"}])
    assert out == "recovered"
    assert route.call_count == 3


@respx.mock
async def test_non_retryable_4xx_raises():
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(401, text="bad key"))
    provider = FriendliProvider(_settings())
    with pytest.raises(ProviderError, match="401"):
        await provider.complete([{"role": "user", "content": "hi"}])


@respx.mock
async def test_malformed_response_raises():
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json={"nope": 1}))
    provider = FriendliProvider(_settings())
    with pytest.raises(ProviderError, match="Unexpected API response"):
        await provider.complete([{"role": "user", "content": "hi"}])

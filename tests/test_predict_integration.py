"""Integration tests for the orchestrator using a fake provider + stub probe."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from oseer.config import Grounding, Settings
from oseer.predict import Predictor
from oseer.providers.base import ProviderError
from oseer.providers.friendli import FriendliProvider
from oseer.schemas import EnvSnapshot, Risk, Source


class FakeProvider:
    """Scripted provider: returns a fixed string, raises, or counts calls."""

    def __init__(self, response: str | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = 0
        self.last_reasoning: bool | None = None

    async def complete(self, messages, reasoning: bool = False):
        self.calls += 1
        self.last_reasoning = reasoning
        if self.error is not None:
            raise self.error
        return self.response or ""


class StubProbe:
    def __init__(self, snap: EnvSnapshot):
        self._snap = snap

    def snapshot(self, cwd=None, refresh=False):
        return self._snap


def _settings() -> Settings:
    return Settings(api_key="x", grounding=Grounding.full, cache_size=16)


def _env(tools=None, shell="/bin/zsh") -> EnvSnapshot:
    return EnvSnapshot(os="macOS", shell=shell, cwd="/proj",
                       tools_available=tools or {"git": True, "node": True})


def _model_json(**overrides) -> str:
    base = {
        "stdout": "file1.txt\nfile2.txt",
        "stderr": "",
        "exit_code": 0,
        "risk": "safe",
        "confidence": 0.8,
        "confidence_basis": "listing known",
    }
    base.update(overrides)
    return json.dumps(base)


def _predictor(provider, env=None, settings=None) -> Predictor:
    return Predictor(
        settings=settings or _settings(),
        provider=provider,
        probe=StubProbe(env or _env()),
    )


async def test_happy_path_returns_model_prediction():
    provider = FakeProvider(_model_json())
    pred = await _predictor(provider).predict_command("ls")
    assert pred.source == Source.model
    assert pred.predicted_stdout.startswith("file1.txt")
    assert pred.risk == Risk.safe
    assert provider.calls == 1


async def test_static_risk_floor_overrides_model_safe():
    # Model naively says 'safe' for a destructive command; the static floor forces destructive.
    provider = FakeProvider(_model_json(risk="safe"))
    pred = await _predictor(provider).predict_command("rm -rf build/")
    assert pred.risk == Risk.destructive
    assert pred.reversible is False
    assert any("deletion" in r for r in pred.risk_reasons)
    assert pred.rollback_hint is not None


async def test_missing_tool_short_circuits_without_calling_model():
    provider = FakeProvider(error=AssertionError("model should NOT be called"))
    env = _env(tools={"docker": False})
    pred = await _predictor(provider, env=env).predict_command("docker ps")
    assert provider.calls == 0
    assert pred.source == Source.static
    assert pred.predicted_exit_code == 127
    assert "docker" in pred.predicted_stderr


async def test_degrades_when_provider_errors():
    provider = FakeProvider(error=ProviderError("network down"))
    pred = await _predictor(provider).predict_command("rm -rf build/")
    assert pred.source == Source.degraded
    # Static risk floor still applies in degraded mode.
    assert pred.risk == Risk.destructive
    assert any("model unavailable" in r for r in pred.risk_reasons)
    assert pred.confidence <= 0.3


async def test_degrades_when_output_unparseable():
    provider = FakeProvider("this is not json")
    pred = await _predictor(provider).predict_command("ls")
    assert pred.source == Source.degraded
    assert pred.confidence <= 0.3


async def test_prediction_is_cached():
    provider = FakeProvider(_model_json())
    predictor = _predictor(provider)
    await predictor.predict_command("ls -la")
    await predictor.predict_command("ls -la")
    assert provider.calls == 1  # second call served from cache


async def test_tool_call_happy_path():
    data = {
        "predicted_result": '{"charged": true}',
        "side_effects": ["POST https://api.stripe/charge"],
        "risk": "destructive",
        "confidence": 0.5,
    }
    provider = FakeProvider(json.dumps(data))
    pred = await _predictor(provider).predict_tool_call("payments", "charge", {"amount": 100})
    assert pred.source == Source.model
    assert pred.risk == Risk.destructive
    assert pred.side_effects


async def test_tool_call_degrades_on_error():
    provider = FakeProvider(error=ProviderError("boom"))
    pred = await _predictor(provider).predict_tool_call("x", "y", {})
    assert pred.source == Source.degraded
    assert pred.risk == Risk.caution


# --- reasoning wiring -----------------------------------------------------------


async def test_reasoning_defaults_to_settings():
    provider = FakeProvider(_model_json())
    settings = Settings(api_key="x", grounding=Grounding.full, reasoning=True)
    await _predictor(provider, settings=settings).predict_command("ls")  # reasoning omitted
    assert provider.last_reasoning is True


async def test_reasoning_per_call_overrides_settings():
    provider = FakeProvider(_model_json())
    settings = Settings(api_key="x", grounding=Grounding.full, reasoning=True)
    await _predictor(provider, settings=settings).predict_command("ls", reasoning=False)
    assert provider.last_reasoning is False


async def test_reasoning_output_with_think_trace_parses_end_to_end():
    # Simulate a real reasoning response: <think> trace then fenced JSON.
    raw = "<think>The dir has two files, ls will list them.</think>\n```json\n" + \
        _model_json(stdout="a.txt\nb.txt") + "\n```"
    provider = FakeProvider(raw)
    pred = await _predictor(provider).predict_command("ls", reasoning=True)
    assert pred.source == Source.model
    assert pred.predicted_stdout == "a.txt\nb.txt"
    assert provider.last_reasoning is True


async def test_reasoning_variant_cached_separately():
    provider = FakeProvider(_model_json())
    predictor = _predictor(provider)
    await predictor.predict_command("ls", reasoning=False)
    await predictor.predict_command("ls", reasoning=True)
    assert provider.calls == 2  # different reasoning → different cache key


# --- full stack over real HTTP (mocked server) ----------------------------------

LOCAL = "http://localhost:8000/v1"


@respx.mock
async def test_full_stack_through_real_provider_over_http():
    """Predictor -> real FriendliProvider -> HTTP -> parse -> merge, keyless self-host."""
    server_json = json.dumps({
        "stdout": "app.py\nbuild/\nREADME.md",
        "stderr": "",
        "exit_code": 0,
        "risk": "safe",
        "confidence": 0.85,
        "confidence_basis": "directory listing is known from the snapshot",
        "assumptions": [],
    })
    route = respx.post(f"{LOCAL}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": server_json}}]})
    )

    settings = Settings(api_base=LOCAL, api_key="", grounding=Grounding.full)  # keyless
    predictor = Predictor(
        settings=settings,
        provider=FriendliProvider(settings),
        probe=StubProbe(_env()),
    )
    pred = await predictor.predict_command("ls")

    assert route.called
    assert "authorization" not in route.calls.last.request.headers  # keyless self-host
    assert pred.source == Source.model
    assert "app.py" in pred.predicted_stdout
    assert pred.risk == Risk.safe
    assert pred.confidence == 0.85
    assert pred.disclaimer  # always present


@respx.mock
async def test_full_stack_destructive_merges_static_floor_over_http():
    # Even if the server underrates risk, the static floor forces destructive.
    server_json = json.dumps({"stdout": "", "stderr": "", "exit_code": 0, "risk": "safe"})
    respx.post(f"{LOCAL}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": server_json}}]})
    )
    settings = Settings(api_base=LOCAL, api_key="", grounding=Grounding.full)
    predictor = Predictor(settings=settings, provider=FriendliProvider(settings),
                          probe=StubProbe(_env()))
    pred = await predictor.predict_command("rm -rf build/")
    assert pred.risk == Risk.destructive
    assert pred.reversible is False


@respx.mock
async def test_full_stack_server_down_degrades_gracefully():
    respx.post(f"{LOCAL}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    settings = Settings(api_base=LOCAL, api_key="", grounding=Grounding.full, retries=0)
    predictor = Predictor(settings=settings, provider=FriendliProvider(settings),
                          probe=StubProbe(_env()))
    pred = await predictor.predict_command("rm -rf build/")
    assert pred.source == Source.degraded
    assert pred.risk == Risk.destructive  # static floor still applies
    assert any("model unavailable" in r for r in pred.risk_reasons)

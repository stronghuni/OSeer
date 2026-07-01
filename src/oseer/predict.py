"""Prediction orchestrator.

Flow for :meth:`Predictor.predict_command`:

    1. Snapshot the (read-only, sanitized) environment.
    2. Static scan → risk assessment. If it yields a deterministic short-circuit
       (e.g. required tool missing), return it WITHOUT calling the model.
    3. Check the prediction cache (keyed by command + environment fingerprint).
    4. Build the world-model prompt and call the provider.
    5. Parse the model output; merge the static risk floor into it.
    6. On any provider/parse failure, degrade gracefully to a static-only prediction.

The static layer is a model-independent safety floor: even when the model is
unreachable or wrong-about-risk, destructive commands stay flagged.
"""

from __future__ import annotations

from collections import OrderedDict

from .config import Settings, get_settings
from .environment import EnvironmentProbe
from .parsing import extract_json_object, to_command_prediction, to_tool_call_prediction
from .prompts import mcp_domain, terminal
from .providers.base import ModelProvider, ProviderError
from .providers.friendli import FriendliProvider
from .safety import StaticAssessment, StaticRiskScanner
from .schemas import (
    CommandPrediction,
    EnvSnapshot,
    Risk,
    Source,
    ToolCallPrediction,
    max_risk,
)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


class _LRU:
    def __init__(self, maxsize: int):
        self._data: OrderedDict[str, object] = OrderedDict()
        self._max = max(1, maxsize)

    def get(self, key: str):
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def put(self, key: str, value: object) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)


class Predictor:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: ModelProvider | None = None,
        probe: EnvironmentProbe | None = None,
    ):
        self._settings = settings or get_settings()
        self._provider = provider or FriendliProvider(self._settings)
        self._probe = probe or EnvironmentProbe(self._settings)
        self._scanner = StaticRiskScanner()
        self._cache = _LRU(self._settings.cache_size)

    # -- terminal domain ------------------------------------------------------

    async def predict_command(
        self,
        command: str,
        cwd: str | None = None,
        shell: str | None = None,
        reasoning: bool | None = None,
    ) -> CommandPrediction:
        env = self._probe.snapshot(cwd=cwd)
        assessment = self._scanner.assess(command, env)

        # (2) deterministic short-circuit — no model call.
        if assessment.short_circuit is not None:
            return assessment.short_circuit

        eff_reasoning = self._settings.reasoning if reasoning is None else reasoning

        # (3) cache (reasoning affects the output, so it is part of the key)
        cache_key = f"cmd::r{int(eff_reasoning)}::{command}::{terminal.env_fingerprint(env)}"
        cached = self._cache.get(cache_key)
        if isinstance(cached, CommandPrediction):
            return cached

        # (4-5) model call + parse, with graceful degradation.
        messages = terminal.build_messages(command, env, self._settings, cwd=cwd, shell=shell)
        try:
            raw = await self._provider.complete(messages, reasoning=eff_reasoning)
        except ProviderError as exc:
            return self._degraded_command(command, assessment, str(exc))

        data = extract_json_object(raw)
        if data is None:
            return self._degraded_command(
                command, assessment, "model output was not parseable as a prediction"
            )

        pred = to_command_prediction(data, command)
        merged = self._merge_command(pred, assessment)
        self._cache.put(cache_key, merged)
        return merged

    def _merge_command(
        self, pred: CommandPrediction, assessment: StaticAssessment
    ) -> CommandPrediction:
        pred.risk = max_risk(pred.risk, assessment.risk)
        pred.risk_reasons = _dedupe(assessment.reasons + pred.risk_reasons)
        pred.reversible = pred.reversible and assessment.reversible
        pred.rollback_hint = assessment.rollback_hint or pred.rollback_hint
        pred.suggestions = _dedupe(assessment.suggestions + pred.suggestions)
        return pred

    def _degraded_command(
        self, command: str, assessment: StaticAssessment, why: str
    ) -> CommandPrediction:
        return CommandPrediction(
            command=command,
            predicted_stderr="",
            predicted_exit_code=0,
            risk=assessment.risk,
            risk_reasons=_dedupe(assessment.reasons + [f"model unavailable: {why}"]),
            reversible=assessment.reversible,
            rollback_hint=assessment.rollback_hint,
            suggestions=assessment.suggestions,
            confidence=0.2 if assessment.reasons else 0.1,
            confidence_basis="Static safety analysis only — the world model was unavailable.",
            assumptions=["Output not predicted; only rule-based risk was assessed."],
            source=Source.degraded,
        )

    # -- mcp domain -----------------------------------------------------------

    async def predict_tool_call(
        self,
        server: str,
        tool: str,
        arguments: dict,
        context: str | None = None,
    ) -> ToolCallPrediction:
        env = self._probe.snapshot()
        messages = mcp_domain.build_messages(
            server, tool, arguments, env, self._settings, context=context
        )
        try:
            raw = await self._provider.complete(messages, reasoning=self._settings.reasoning)
        except ProviderError as exc:
            return self._degraded_tool_call(server, tool, str(exc))

        data = extract_json_object(raw)
        if data is None:
            return self._degraded_tool_call(
                server, tool, "model output was not parseable as a prediction"
            )
        return to_tool_call_prediction(data, server, tool)

    def _degraded_tool_call(self, server: str, tool: str, why: str) -> ToolCallPrediction:
        return ToolCallPrediction(
            server=server,
            tool=tool,
            predicted_result="",
            risk=Risk.caution,
            risk_reasons=[f"model unavailable: {why}"],
            confidence=0.1,
            confidence_basis="The world model was unavailable; no prediction could be made.",
            assumptions=["Unknown tool behavior; treat as uncertain."],
            source=Source.degraded,
        )

    # -- env transparency -----------------------------------------------------

    def env_snapshot(self, refresh: bool = False) -> EnvSnapshot:
        return self._probe.snapshot(refresh=refresh)

"""Parse the world model's raw text into structured predictions.

The model is asked for a single JSON object, but real outputs drift: fenced code blocks,
a leading ``<think>`` block, or trailing prose. This module is tolerant — it strips
reasoning, finds the JSON, and coerces fields into the Pydantic schema, degrading
gracefully rather than raising when a field is malformed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .schemas import CommandPrediction, Risk, Source, ToolCallPrediction

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` (and an unclosed trailing ``<think>``)."""
    text = _THINK_RE.sub("", text)
    # Unclosed <think> — drop everything up to it if a JSON object follows later.
    if "<think>" in text.lower():
        idx = text.lower().index("<think>")
        after = text[idx + len("<think>"):]
        if "{" in after:
            text = after
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in ``text`` (fenced or bare), or None."""
    cleaned = strip_reasoning(text)

    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        obj = _try_load(fenced.group(1))
        if obj is not None:
            return obj

    # Bare object: scan for the first balanced {...}.
    for candidate in _balanced_objects(cleaned):
        obj = _try_load(candidate)
        if obj is not None:
            return obj
    return None


def _try_load(blob: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _balanced_objects(text: str):
    """Yield substrings that are balanced-brace candidates, largest-first at each start."""
    starts = [i for i, c in enumerate(text) if c == "{"]
    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break


# --- coercion helpers -------------------------------------------------------


def _as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _as_str_list(v: Any) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [_as_str(x) for x in v if x not in (None, "")]
    return [_as_str(v)]


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_bool(v: Any, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return default


def _as_float01(v: Any, default: float = 0.5) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, f))


def _risk_from_number(n: float) -> Risk:
    """Map an ordinal 0/1/2 risk scale (some models emit numbers) to the enum."""
    i = int(n)
    if i <= 0:
        return Risk.safe
    if i == 1:
        return Risk.caution
    return Risk.destructive


def _as_risk(v: Any) -> Risk:
    if isinstance(v, bool):  # guard: bool is a subclass of int
        return Risk.destructive if v else Risk.safe
    if isinstance(v, (int, float)):
        return _risk_from_number(v)
    if isinstance(v, str):
        key = v.strip().lower()
        if key in Risk._value2member_map_:
            return Risk(key)
        if key.replace(".", "", 1).isdigit():  # numeric string, e.g. "2"
            return _risk_from_number(float(key))
        if key in {"high", "danger", "dangerous", "critical", "severe"}:
            return Risk.destructive
        if key in {"medium", "moderate", "warn", "warning", "low", "minimal"}:
            return Risk.caution
    return Risk.safe


def _rollback(v: Any) -> str | None:
    s = _as_str(v).strip()
    if not s or s.lower() in {"null", "none", "n/a"}:
        return None
    return s


def to_command_prediction(data: dict[str, Any], command: str) -> CommandPrediction:
    return CommandPrediction(
        command=command,
        predicted_stdout=_as_str(data.get("stdout")),
        predicted_stderr=_as_str(data.get("stderr")),
        predicted_exit_code=_as_int(data.get("exit_code")),
        filesystem_effects=_as_str_list(data.get("filesystem_effects")),
        state_changes=_as_str_list(data.get("state_changes")),
        risk=_as_risk(data.get("risk")),
        risk_reasons=_as_str_list(data.get("risk_reasons")),
        reversible=_as_bool(data.get("reversible")),
        rollback_hint=_rollback(data.get("rollback_hint")),
        suggestions=_as_str_list(data.get("suggestions")),
        confidence=_as_float01(data.get("confidence")),
        confidence_basis=_as_str(data.get("confidence_basis")),
        assumptions=_as_str_list(data.get("assumptions")),
        source=Source.model,
    )


def to_tool_call_prediction(data: dict[str, Any], server: str, tool: str) -> ToolCallPrediction:
    return ToolCallPrediction(
        server=server,
        tool=tool,
        predicted_result=_as_str(data.get("predicted_result")),
        side_effects=_as_str_list(data.get("side_effects")),
        risk=_as_risk(data.get("risk")),
        risk_reasons=_as_str_list(data.get("risk_reasons")),
        reversible=_as_bool(data.get("reversible")),
        rollback_hint=_rollback(data.get("rollback_hint")),
        suggestions=_as_str_list(data.get("suggestions")),
        confidence=_as_float01(data.get("confidence")),
        confidence_basis=_as_str(data.get("confidence_basis")),
        assumptions=_as_str_list(data.get("assumptions")),
        source=Source.model,
    )

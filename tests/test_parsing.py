"""Offline tests for output parsing and field coercion."""

from __future__ import annotations

from oseer.parsing import (
    extract_json_object,
    strip_reasoning,
    to_command_prediction,
    to_tool_call_prediction,
)
from oseer.schemas import Risk, Source

CLEAN = '{"stdout": "ok", "stderr": "", "exit_code": 0, "risk": "safe", "confidence": 0.8}'


def test_strip_reasoning_removes_think_block():
    text = "<think>let me reason\nabout this</think>\n{\"exit_code\": 0}"
    assert "<think>" not in strip_reasoning(text)
    assert '{"exit_code": 0}' in strip_reasoning(text)


def test_extract_plain_json():
    obj = extract_json_object(CLEAN)
    assert obj is not None
    assert obj["exit_code"] == 0


def test_extract_json_from_fenced_block():
    text = "Here is my prediction:\n```json\n" + CLEAN + "\n```\nDone."
    obj = extract_json_object(text)
    assert obj is not None
    assert obj["risk"] == "safe"


def test_extract_json_after_think_and_prose():
    text = "<think>reasoning here</think>\nPrediction:\n" + CLEAN
    obj = extract_json_object(text)
    assert obj is not None
    assert obj["stdout"] == "ok"


def test_extract_handles_nested_braces():
    text = '{"stdout": "{nested: value}", "state_changes": ["a=1"], "exit_code": 0}'
    obj = extract_json_object(text)
    assert obj is not None
    assert obj["state_changes"] == ["a=1"]


def test_extract_returns_none_on_garbage():
    assert extract_json_object("no json here at all") is None
    assert extract_json_object("") is None


def test_to_command_prediction_full():
    data = {
        "stdout": "file1\nfile2",
        "stderr": "",
        "exit_code": 0,
        "filesystem_effects": ["created out.txt"],
        "state_changes": [],
        "risk": "caution",
        "risk_reasons": ["writes a file"],
        "reversible": False,
        "rollback_hint": "rm out.txt",
        "suggestions": ["use --dry-run"],
        "confidence": 0.72,
        "confidence_basis": "listing is known",
        "assumptions": ["cwd unchanged"],
    }
    pred = to_command_prediction(data, "ls > out.txt")
    assert pred.command == "ls > out.txt"
    assert pred.predicted_stdout == "file1\nfile2"
    assert pred.risk == Risk.caution
    assert pred.reversible is False
    assert pred.rollback_hint == "rm out.txt"
    assert pred.confidence == 0.72
    assert pred.source == Source.model


def test_coercion_tolerates_wrong_types():
    data = {
        "stdout": 123,                    # int -> str
        "exit_code": "1",                # str -> int
        "risk": "DANGER",                # alias -> destructive
        "reversible": "false",           # str -> bool
        "risk_reasons": "single reason",  # str -> [str]
        "confidence": "1.7",             # clamp to 1.0
        "rollback_hint": "null",         # -> None
    }
    pred = to_command_prediction(data, "x")
    assert pred.predicted_stdout == "123"
    assert pred.predicted_exit_code == 1
    assert pred.risk == Risk.destructive
    assert pred.reversible is False
    assert pred.risk_reasons == ["single reason"]
    assert pred.confidence == 1.0
    assert pred.rollback_hint is None


def test_numeric_risk_scale():
    # Some models emit an ordinal 0/1/2 risk scale instead of the string enum.
    assert to_command_prediction({"risk": 0}, "x").risk == Risk.safe
    assert to_command_prediction({"risk": 1}, "x").risk == Risk.caution
    assert to_command_prediction({"risk": 2}, "x").risk == Risk.destructive
    assert to_command_prediction({"risk": 3}, "x").risk == Risk.destructive
    assert to_command_prediction({"risk": "2"}, "x").risk == Risk.destructive
    assert to_command_prediction({"risk": "high"}, "x").risk == Risk.destructive


def test_missing_fields_get_defaults():
    pred = to_command_prediction({}, "echo hi")
    assert pred.predicted_stdout == ""
    assert pred.predicted_exit_code == 0
    assert pred.risk == Risk.safe
    assert pred.filesystem_effects == []


def test_to_tool_call_prediction():
    data = {
        "predicted_result": '{"ok": true}',
        "side_effects": ["POST to https://api.example.com"],
        "risk": "destructive",
        "confidence": 0.4,
    }
    pred = to_tool_call_prediction(data, "payments", "charge_card")
    assert pred.server == "payments"
    assert pred.tool == "charge_card"
    assert pred.risk == Risk.destructive
    assert pred.side_effects == ["POST to https://api.example.com"]

"""MCP-domain world-model prompt construction — predict the observation of a tool call."""

from __future__ import annotations

import json

from ..config import Settings
from ..providers.base import Message
from ..schemas import EnvSnapshot

_OUTPUT_CONTRACT = """\
Respond with a SINGLE JSON object (no prose outside it) predicting the result of the tool \
call being made ONCE. Use exactly these keys:

{
  "predicted_result": "<what the tool would most likely return, as text or JSON>",
  "side_effects": ["<external state changes: files, network, DB rows, remote resources>"],
  "risk": "safe | caution | destructive",
  "risk_reasons": ["<why, if not safe>"],
  "reversible": <true|false>,
  "rollback_hint": "<how to undo, or null>",
  "suggestions": ["<safer or more efficient alternatives, if any>"],
  "confidence": <0.0-1.0>,
  "confidence_basis": "<what makes this prediction more/less certain>",
  "assumptions": ["<anything you assumed about the tool/server/state>"]
}

Rules:
- Reason about what a tool named like this, with these arguments, most plausibly does.
- Treat writes, deletes, sends, and payments as caution/destructive as appropriate.
- If the tool's behavior is genuinely unknowable, say so via low confidence + assumptions.
"""


def build_messages(
    server: str,
    tool: str,
    arguments: dict,
    env: EnvSnapshot | None,
    settings: Settings,
    context: str | None = None,
) -> list[Message]:
    system = (
        "You are a language world model simulating an MCP (Model Context Protocol) tool "
        "environment. Given a tool call, predict the observation the tool would return and any "
        "side effects, without executing it.\n\n"
        f"{_OUTPUT_CONTRACT}"
    )

    try:
        args_str = json.dumps(arguments, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        args_str = repr(arguments)

    user_parts = [
        f"Action: call_mcp_tool",
        f"Server: {server}",
        f"Tool: {tool}",
        f"Arguments: {args_str}",
    ]
    if context:
        user_parts.append(f"Context: {context}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

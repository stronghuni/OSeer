#!/usr/bin/env python3
"""Live smoke test against the real model API.

Requires OSEER_API_KEY (or FRIENDLI_TOKEN). Exercises a spread of commands:
a safe one, a destructive one, a force-push, and a command using a tool that is
NOT installed (which should short-circuit WITHOUT an API call).

Usage:
    uv run python scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import sys

from oseer.config import get_settings
from oseer.predict import Predictor

COMMANDS = [
    "ls -la",
    "rm -rf build/",
    "git push --force origin main",
    "definitely-not-a-real-tool --version",  # expect static short-circuit, exit 127
]


def _fmt(pred) -> str:
    lines = [
        f"  source:     {pred.source.value}",
        f"  risk:       {pred.risk.value}   reversible={pred.reversible}",
        f"  exit_code:  {pred.predicted_exit_code}",
        f"  confidence: {pred.confidence:.2f}  ({pred.confidence_basis})",
    ]
    if pred.predicted_stdout:
        head = pred.predicted_stdout.splitlines()[:4]
        lines.append("  stdout:     " + " / ".join(head))
    if pred.predicted_stderr:
        lines.append(f"  stderr:     {pred.predicted_stderr.splitlines()[0]}")
    if pred.risk_reasons:
        lines.append("  reasons:    " + "; ".join(pred.risk_reasons))
    if pred.rollback_hint:
        lines.append(f"  rollback:   {pred.rollback_hint}")
    if pred.suggestions:
        lines.append("  suggest:    " + "; ".join(pred.suggestions))
    return "\n".join(lines)


async def main() -> int:
    settings = get_settings()
    auth = "with key" if settings.api_key_present() else "keyless (self-hosted)"

    print(f"Model:    {settings.model}")
    print(f"Endpoint: {settings.api_base}  [{auth}]")
    print(f"Grounding:{settings.grounding.value}")
    print("If the model server is unreachable, predictions degrade to static-only.\n")

    predictor = Predictor(settings)
    for cmd in COMMANDS:
        print(f"$ {cmd}")
        try:
            pred = await predictor.predict_command(cmd)
            print(_fmt(pred))
        except Exception as exc:  # noqa: BLE001 — smoke script surfaces anything
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        print()

    print("Note: predictions are advisory, not execution. Accuracy is imperfect.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

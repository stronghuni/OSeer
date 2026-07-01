#!/usr/bin/env python3
"""Meticulous end-to-end verification against the live model server.

Exercises every feature layer with explicit PASS/FAIL assertions:
env grounding, secret sanitization, all risk tiers, the static short-circuit,
reasoning mode, caching, and graceful degradation. Exits non-zero on any failure.

Usage:
    OSEER_API_BASE=http://host:port/v1 uv run python scripts/verify_live.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from oseer.config import Grounding, Settings, get_settings
from oseer.environment import EnvironmentProbe
from oseer.predict import Predictor
from oseer.providers.friendli import FriendliProvider
from oseer.schemas import EnvSnapshot, Risk, Source

PASS, FAIL = "✅ PASS", "❌ FAIL"
_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((ok, name, detail))
    print(f"  {PASS if ok else FAIL}  {name}" + (f"  — {detail}" if detail else ""))


class StubProbe:
    def __init__(self, snap: EnvSnapshot):
        self._snap = snap

    def snapshot(self, cwd=None, refresh=False):
        return self._snap


class CountingProvider(FriendliProvider):
    def __init__(self, settings):
        super().__init__(settings)
        self.calls = 0

    async def complete(self, messages, reasoning=False):
        self.calls += 1
        return await super().complete(messages, reasoning=reasoning)


async def main() -> int:
    base = get_settings()
    print(f"Endpoint: {base.api_base}  |  model: {base.model}\n")

    # --- 1. environment grounding -------------------------------------------
    print("1. Environment grounding")
    full = EnvironmentProbe(Settings(grounding=Grounding.full)).snapshot()
    check("full grounding captures OS + tools", bool(full.os) and bool(full.tools_available),
          f"os={full.os!r}, tools={len(full.tools_available)}")
    none = EnvironmentProbe(Settings(grounding=Grounding.none)).snapshot()
    check("grounding=none sends nothing", none.os == "" and none.tools_available == {})

    # --- 2. secret sanitization (never leaks to the server) -----------------
    print("\n2. Secret sanitization")
    os.environ["AWS_SECRET_ACCESS_KEY"] = "LEAK_should_never_appear"
    snap = EnvironmentProbe(Settings(grounding=Grounding.full)).snapshot()
    blob = repr(snap.model_dump())
    check("secret value absent from snapshot", "LEAK_should_never_appear" not in blob)

    # --- 3. terminal predictions: every risk tier (live model) --------------
    print("\n3. Terminal predictions (live model)")
    p = Predictor(base)
    safe = await p.predict_command("ls -la")
    check("safe cmd → model, exit 0, risk safe",
          safe.source == Source.model and safe.predicted_exit_code == 0 and safe.risk == Risk.safe,
          f"src={safe.source.value}, exit={safe.predicted_exit_code}, risk={safe.risk.value}")

    dest = await p.predict_command("rm -rf /tmp/oseer_demo_dir")
    check("destructive cmd → risk destructive + irreversible",
          dest.risk == Risk.destructive and dest.reversible is False and bool(dest.rollback_hint),
          f"risk={dest.risk.value}, reversible={dest.reversible}")

    force = await p.predict_command("git push --force origin main")
    check("force push → destructive", force.risk == Risk.destructive)

    # --- 4. deterministic short-circuit (no model call) ---------------------
    print("\n4. Static short-circuit (missing tool, no model call)")
    env_missing = EnvSnapshot(os="Ubuntu", shell="/bin/bash",
                              tools_available={"docker": False, "git": True})
    counter = CountingProvider(base)
    sc_pred = await Predictor(base, provider=counter, probe=StubProbe(env_missing)).predict_command("docker ps")
    check("missing tool → static source, exit 127, zero model calls",
          sc_pred.source == Source.static and sc_pred.predicted_exit_code == 127 and counter.calls == 0,
          f"src={sc_pred.source.value}, exit={sc_pred.predicted_exit_code}, model_calls={counter.calls}")

    # --- 5. reasoning mode (<think> stripped) -------------------------------
    print("\n5. Reasoning mode")
    r = await p.predict_command("echo reasoning-check", reasoning=True)
    joined = r.predicted_stdout + r.predicted_stderr + " ".join(r.risk_reasons)
    check("reasoning=True → valid prediction, no <think> leakage",
          r.source == Source.model and "<think>" not in joined,
          f"src={r.source.value}, stdout={r.predicted_stdout[:30]!r}")

    # --- 6. caching (identical dry-run served from cache) -------------------
    print("\n6. Prediction cache")
    counter2 = CountingProvider(base)
    pc = Predictor(base, provider=counter2)  # real probe
    await pc.predict_command("uname -a")
    await pc.predict_command("uname -a")
    check("identical command → one model call", counter2.calls == 1, f"model_calls={counter2.calls}")

    # --- 7. graceful degradation (dead endpoint) ----------------------------
    print("\n7. Graceful degradation (dead endpoint)")
    dead = Settings(api_base="http://127.0.0.1:1/v1", retries=0, timeout=2, grounding=Grounding.full)
    dp = await Predictor(dead).predict_command("rm -rf build/")
    check("dead server → degraded but destructive floor holds",
          dp.source == Source.degraded and dp.risk == Risk.destructive and dp.confidence <= 0.3,
          f"src={dp.source.value}, risk={dp.risk.value}, conf={dp.confidence}")

    # --- 8. MCP tool-call predictions (live model) --------------------------
    print("\n8. MCP tool-call predictions (live model)")
    read_pred = await p.predict_tool_call("filesystem", "read_file", {"path": "/etc/hostname"})
    check("read tool → prediction returned", read_pred.source == Source.model,
          f"risk={read_pred.risk.value}, conf={read_pred.confidence}")
    charge_pred = await p.predict_tool_call("payments", "charge_card",
                                            {"amount": 5000, "currency": "USD"},
                                            context="Charges a customer's card immediately.")
    check("payment tool → caution/destructive",
          charge_pred.risk in (Risk.caution, Risk.destructive),
          f"risk={charge_pred.risk.value}")

    # --- summary ------------------------------------------------------------
    passed = sum(1 for ok, *_ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 60)
    print(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

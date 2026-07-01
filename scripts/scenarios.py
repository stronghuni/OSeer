#!/usr/bin/env python3
"""Diverse-environment scenario matrix against the live model.

Runs the SAME commands across several synthetic user "personas" (different OS, shell,
installed tools, git state) to show OSeer adapts predictions to each environment — and
verifies invariants that must hold regardless of what the model says (static risk floor,
environment-aware short-circuits).

Usage:
    OSEER_API_BASE=http://host:port/v1 uv run python scripts/scenarios.py
"""

from __future__ import annotations

import asyncio
import sys

from oseer.config import get_settings
from oseer.predict import Predictor
from oseer.schemas import EnvSnapshot, Risk, Source


class StubProbe:
    def __init__(self, snap: EnvSnapshot):
        self._snap = snap

    def snapshot(self, cwd=None, refresh=False):
        return self._snap


def _tools(present: list[str], absent: list[str]) -> dict[str, bool]:
    return {**{t: True for t in present}, **{t: False for t in absent}}


# --- personas -------------------------------------------------------------------

PERSONAS: dict[str, EnvSnapshot] = {
    "macOS/zsh dev": EnvSnapshot(
        os="macOS", os_version="26.5", shell="/bin/zsh", cwd="/Users/dev/app",
        cwd_listing=["package.json", "node_modules/", "src/", ".git/"],
        git="branch=main clean origin=git@github.com:dev/app.git ahead=0 behind=0",
        tools_available=_tools(["git", "node", "npm", "brew", "docker", "python3"],
                               ["apt-get", "apk"]),
        package_managers=["brew", "npm"],
    ),
    "Ubuntu/bash server": EnvSnapshot(
        os="Ubuntu", os_version="22.04", shell="/bin/bash", cwd="/var/www/app",
        cwd_listing=["app.py", "requirements.txt", ".git/"],
        git="branch=main 2 uncommitted change(s) origin=https://github.com/org/app.git ahead=0 behind=3",
        tools_available=_tools(["git", "apt-get", "docker", "python3", "curl"],
                               ["brew", "node", "apk"]),
        package_managers=["apt-get", "pip"],
    ),
    "Alpine container": EnvSnapshot(
        os="Alpine Linux", os_version="3.19", shell="/bin/sh", cwd="/app",
        cwd_listing=["main.go", "go.mod"],
        git=None,
        tools_available=_tools(["apk", "go"],
                               ["git", "brew", "apt-get", "node", "docker", "python3"]),
        package_managers=["apk"],
    ),
    "Python venv (macOS)": EnvSnapshot(
        os="macOS", os_version="26.5", shell="/bin/zsh", cwd="/Users/dev/pyproj",
        cwd_listing=["pyproject.toml", ".venv/", "src/"],
        git="branch=feature clean origin=git@github.com:dev/pyproj.git no-upstream",
        tools_available=_tools(["git", "python3", "uv", "brew"], ["node", "apt-get", "docker"]),
        package_managers=["uv", "pip"],
        env_vars={"VIRTUAL_ENV": "/Users/dev/pyproj/.venv"},
    ),
    "No-git machine": EnvSnapshot(
        os="Debian", os_version="12", shell="/bin/bash", cwd="/root",
        cwd_listing=["notes.txt"],
        git=None,
        tools_available=_tools(["apt-get", "curl"], ["git", "brew", "node", "docker"]),
        package_managers=["apt-get"],
    ),
}

# Same commands run across every persona → predictions should differ by environment.
COMMANDS = [
    "ls -la",
    "brew install jq",
    "apt-get install -y curl",
    "git push --force origin main",
    "rm -rf node_modules",
    "docker compose up -d",
]

# Invariants that must hold no matter what the model returns.
def _check_invariants(cmd: str, persona: str, env: EnvSnapshot, pred) -> list[str]:
    fails = []
    if "rm -rf" in cmd or "push --force" in cmd:
        if pred.risk != Risk.destructive:
            fails.append(f"{persona}: '{cmd}' should be destructive, got {pred.risk.value}")
    # Environment-aware short-circuit: a first-token tool marked absent → 127 without model.
    first = cmd.split()[0]
    if env.tools_available.get(first) is False:
        if pred.predicted_exit_code != 127:
            fails.append(f"{persona}: '{cmd}' expected exit 127 (missing {first}), "
                         f"got {pred.predicted_exit_code}")
    return fails


def _row(persona: str, cmd: str, pred) -> str:
    src = pred.source.value[:5]
    out = (pred.predicted_stdout or pred.predicted_stderr or "").replace("\n", " ")[:44]
    return (f"  {cmd:<32} {pred.risk.value:<11} exit={pred.predicted_exit_code:<4} "
            f"{src:<6} c={pred.confidence:.2f}  {out}")


async def main() -> int:
    s = get_settings()
    print(f"Endpoint: {s.api_base}  |  model: {s.model}\n")
    predictor_cache: dict[str, Predictor] = {}
    all_fails: list[str] = []

    for persona, env in PERSONAS.items():
        print(f"### {persona}  [{env.os} {env.shell}]")
        predictor = Predictor(settings=s, probe=StubProbe(env))
        for cmd in COMMANDS:
            try:
                pred = await predictor.predict_command(cmd)
            except Exception as exc:  # noqa: BLE001
                print(f"  {cmd:<32} ERROR: {type(exc).__name__}: {exc}")
                continue
            print(_row(persona, cmd, pred))
            all_fails += _check_invariants(cmd, persona, env, pred)
        print()

    print("=" * 70)
    if all_fails:
        print(f"INVARIANT FAILURES ({len(all_fails)}):")
        for f in all_fails:
            print("  ✗", f)
        return 1
    total = len(PERSONAS) * len(COMMANDS)
    print(f"All invariants held across {total} predictions "
          f"({len(PERSONAS)} personas × {len(COMMANDS)} commands).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

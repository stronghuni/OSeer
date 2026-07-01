"""Terminal-domain world-model prompt construction."""

from __future__ import annotations

import json

from ..config import Settings
from ..providers.base import Message
from ..schemas import EnvSnapshot

# The output contract. We ask for a single JSON object so the result is machine-parseable,
# while still letting the model reason first (in <think> when enabled).
_OUTPUT_CONTRACT = """\
Respond with a SINGLE JSON object (no prose outside it) describing the predicted result of \
running the command ONCE in the environment above. Use exactly these keys:

{
  "stdout": "<predicted standard output, verbatim>",
  "stderr": "<predicted standard error, verbatim>",
  "exit_code": <integer>,
  "filesystem_effects": ["<files created/modified/deleted, if any>"],
  "state_changes": ["<env vars, cwd, installed packages, processes, etc.>"],
  "risk": "safe | caution | destructive",
  "risk_reasons": ["<why, if not safe>"],
  "reversible": <true|false>,
  "rollback_hint": "<how to undo, or null>",
  "suggestions": ["<safer or more efficient alternatives, if any>"],
  "confidence": <0.0-1.0>,
  "confidence_basis": "<what makes this prediction more/less certain>",
  "assumptions": ["<anything you assumed about unknown state>"]
}

Rules:
- Predict the MOST LIKELY real outcome for THIS machine, not a generic one.
- If the command would fail, put the realistic error in stderr and a non-zero exit_code.
- Do not actually invent success for commands whose required tools are missing.
- Keep stdout/stderr realistic but truncate extremely long output with a note.
"""


def _format_env(env: EnvSnapshot | None) -> str:
    if env is None or env.grounding == "none":
        return "Environment: (not provided — predict conservatively and note assumptions)."

    lines = ["Environment (real, read-only snapshot of the user's machine):"]
    if env.os:
        lines.append(f"- OS: {env.os} {env.os_version} (kernel {env.kernel})")
    if env.shell:
        lines.append(f"- Shell: {env.shell} {env.shell_version}".rstrip())
    if env.cwd:
        lines.append(f"- Working directory: {env.cwd}")
    if env.cwd_listing:
        lines.append(f"- Directory contents: {', '.join(env.cwd_listing)}")
    if env.git:
        lines.append(f"- Git: {env.git}")
    if env.tools_available:
        present = sorted(t for t, ok in env.tools_available.items() if ok)
        missing = sorted(t for t, ok in env.tools_available.items() if not ok)
        lines.append(f"- Tools present: {', '.join(present) or '(none of the checked set)'}")
        if missing:
            lines.append(f"- Tools NOT installed: {', '.join(missing)}")
    if env.package_managers:
        lines.append(f"- Package managers: {', '.join(env.package_managers)}")
    if env.env_vars:
        rendered = ", ".join(f"{k}={v}" for k, v in env.env_vars.items())
        lines.append(f"- Env vars (secrets redacted): {rendered}")
    return "\n".join(lines)


def build_messages(
    command: str,
    env: EnvSnapshot | None,
    settings: Settings,
    cwd: str | None = None,
    shell: str | None = None,
) -> list[Message]:
    os_name = (env.os if env and env.os else "Unix")
    shell_name = shell or (env.shell if env else "") or "sh"

    system = (
        f"You are a language world model simulating a {os_name} terminal running {shell_name}. "
        "Given the user's command and the environment state, predict the terminal's observation "
        "— what the shell would output and how machine state would change — without executing "
        "anything. Base predictions on the specific environment provided.\n\n"
        f"{_format_env(env)}\n\n"
        f"{_OUTPUT_CONTRACT}"
    )

    action = f"Action: execute_bash\nCommand: {command}"
    if cwd:
        action += f"\n(Working directory: {cwd})"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": action},
    ]


def env_fingerprint(env: EnvSnapshot | None) -> str:
    """Stable string used (with the command) as the prediction cache key."""
    if env is None:
        return "none"
    payload = {
        "os": env.os,
        "os_version": env.os_version,
        "shell": env.shell,
        "cwd": env.cwd,
        "listing": env.cwd_listing,
        "git": env.git,
        "tools": env.tools_available,
        "grounding": env.grounding,
    }
    return json.dumps(payload, sort_keys=True)

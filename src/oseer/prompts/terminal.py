"""Terminal-domain world-model prompt construction."""

from __future__ import annotations

import json
import os
import re
import shlex

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
- "risk" MUST be exactly one of the strings: "safe", "caution", "destructive" (not a number).
- Ground every field in the facts above: the OS/shell, the directory contents, the git state,
  and which tools are/aren't installed. Do not contradict them.
- If a required tool is listed as NOT installed, predict "command not found" (exit 127) — never
  fabricate success.
- If the command targets a path, use the "Referenced paths" facts: a missing path means the
  command fails (or no-ops for rm/rmdir) — do not invent files that aren't there.
- If the command would fail, put the realistic error in stderr and a non-zero exit_code.
- Keep stdout/stderr realistic but truncate extremely long output with a "... (truncated)" note.
- Set "confidence" honestly: high when fully determined by the facts, low when you had to guess.
"""


# Path-like tokens whose existence we can verify cheaply (huge accuracy win for rm/cd/cat/cp...).
_PATH_TOKEN = re.compile(r"^[~./]|/")


def _referenced_paths(command: str, cwd: str | None) -> str | None:
    """Report which path-like arguments in the command actually exist under cwd.

    Only runs when ``cwd`` is a real directory on THIS machine — otherwise (synthetic or
    remote environments) we cannot verify paths and must not fabricate existence facts.
    """
    base = cwd or os.getcwd()
    if not os.path.isdir(base):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    seen: list[str] = []
    findings: list[str] = []
    for tok in tokens[1:]:  # skip the program itself
        if tok.startswith("-") or "=" in tok[:1]:
            continue
        if not _PATH_TOKEN.search(tok) and tok not in ("build", "dist", "node_modules"):
            # only judge things that look like paths (or very common dir names)
            if "/" not in tok and "." not in tok:
                continue
        if tok in seen:
            continue
        seen.append(tok)
        expanded = os.path.expanduser(tok)
        full = expanded if os.path.isabs(expanded) else os.path.join(base, expanded)
        if os.path.isdir(full):
            findings.append(f"{tok} (exists: directory)")
        elif os.path.exists(full):
            findings.append(f"{tok} (exists: file)")
        else:
            findings.append(f"{tok} (does NOT exist)")
        if len(findings) >= 8:
            break
    if not findings:
        return None
    return "Referenced paths: " + "; ".join(findings)


def _program_note(command: str, env: EnvSnapshot | None) -> str | None:
    """State whether the command's own program is installed — the model otherwise sometimes
    ignores the tools list and hallucinates 'command not found' for a present tool."""
    if env is None or not env.tools_available:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for tok in tokens:
        if tok == "sudo" or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            continue
        if "/" in tok:  # explicit path, not a PATH lookup
            return None
        installed = env.tools_available.get(tok)
        if installed is True:
            return (f"IMPORTANT: the command's program '{tok}' IS installed and on PATH — "
                    "do not predict 'command not found' for it.")
        if installed is False:
            return (f"IMPORTANT: the command's program '{tok}' is NOT installed — predict "
                    f"'{tok}: command not found' with exit code 127.")
        return None
    return None


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

    env_block = _format_env(env)
    if env is not None and env.grounding != "none":
        note = _program_note(command, env)
        if note:
            env_block += "\n" + note
        # Path verification only against the real local machine (not synthetic/none).
        refs = _referenced_paths(command, cwd or (env.cwd if env else None))
        if refs:
            env_block += "\n" + refs

    system = (
        f"You are a language world model simulating a {os_name} terminal running {shell_name}. "
        "Given the user's command and the environment state, predict the terminal's observation "
        "— what the shell would output and how machine state would change — without executing "
        "anything. Base predictions on the specific environment provided.\n\n"
        f"{env_block}\n\n"
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

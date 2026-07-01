"""Fast, model-independent safety scanning of shell commands.

Two jobs:

1. **Risk assessment** — a rule pass over the command string that flags destructive
   patterns (``rm -rf``, ``git push --force``, ``curl | sh`` …), with reasons,
   reversibility, rollback hints, and safer alternatives. This is the model-independent
   safety floor: it runs regardless of whether the world model is reachable.

2. **Deterministic short-circuit** — using the environment snapshot, catch outcomes we
   can predict for free (e.g. the first token isn't on ``PATH`` → "command not found").
   These skip the model call entirely, saving latency and API cost.

Everything is advisory. OSeer never blocks a command; it annotates it.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from .schemas import CommandPrediction, EnvSnapshot, Risk, Source, max_risk


@dataclass
class RiskRule:
    pattern: re.Pattern[str]
    risk: Risk
    reason: str
    reversible: bool = True
    rollback: str | None = None
    suggestion: str | None = None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Ordered rules; every match contributes a reason. Severity is merged with ``max_risk``.
RISK_RULES: list[RiskRule] = [
    RiskRule(
        _rx(r"\brm\s+(-[a-z]*\s+)*-[a-z]*[rf][a-z]*"),
        Risk.destructive,
        "recursive/forced file deletion (rm -r/-f)",
        reversible=False,
        rollback="No rollback — deleted files are gone unless externally backed up.",
        suggestion="Scope the path explicitly, add -i for confirmation, or move to a trash dir instead.",
    ),
    RiskRule(
        _rx(r"\brm\b.*(\s|/)(/|\$HOME|~)(\s|$)"),
        Risk.destructive,
        "deletion targeting a root or home path",
        reversible=False,
        rollback="No rollback.",
        suggestion="Double-check the target path; never rm at / or ~ scope.",
    ),
    RiskRule(
        _rx(r"\b(dd)\b.*\bof=/dev/"),
        Risk.destructive,
        "dd writing directly to a device (can wipe a disk)",
        reversible=False,
        rollback="No rollback — raw device writes are unrecoverable.",
    ),
    RiskRule(
        _rx(r"\bmkfs(\.\w+)?\b|\bfdisk\b|\bdiskutil\s+(erase|partition)"),
        Risk.destructive,
        "filesystem/partition formatting",
        reversible=False,
        rollback="No rollback — formatting destroys existing data.",
    ),
    RiskRule(
        _rx(r":\(\)\s*\{.*\}\s*;\s*:|\bfork\s*bomb"),
        Risk.destructive,
        "fork bomb / resource exhaustion",
        reversible=False,
    ),
    RiskRule(
        _rx(r"\bgit\s+push\b.*(--force(?!-with-lease)|-f\b)"),
        Risk.destructive,
        "force push can overwrite remote history for others",
        reversible=False,
        rollback="Recoverable only via reflog if someone still has the old commits.",
        suggestion="Use --force-with-lease, or push to a new branch.",
    ),
    RiskRule(
        _rx(r"\bgit\s+reset\s+--hard\b"),
        Risk.destructive,
        "git reset --hard discards uncommitted changes",
        reversible=False,
        rollback="Committed work may be recoverable via `git reflog`; uncommitted work is lost.",
        suggestion="Stash first: `git stash` — then reset.",
    ),
    RiskRule(
        _rx(r"\bgit\s+clean\s+-[a-z]*f"),
        Risk.destructive,
        "git clean -f removes untracked files permanently",
        reversible=False,
        rollback="No rollback for removed untracked files.",
        suggestion="Preview with `git clean -n` first.",
    ),
    RiskRule(
        _rx(r"\bcurl\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh)\b|\bwget\b[^|]*\|\s*(sudo\s+)?(sh|bash)"),
        Risk.destructive,
        "piping a downloaded script straight into a shell (remote code execution)",
        reversible=False,
        suggestion="Download, read, then run: `curl -fsSL URL -o script.sh` and inspect it first.",
    ),
    RiskRule(
        _rx(r"\bchmod\s+(-[a-z]*R[a-z]*\s+)?0*777\b|\bchmod\s+-R\b"),
        Risk.caution,
        "broad/recursive permission change",
        suggestion="Grant the narrowest permissions needed instead of 777.",
    ),
    RiskRule(
        _rx(r"\bchown\s+-R\b"),
        Risk.caution,
        "recursive ownership change",
    ),
    RiskRule(
        _rx(r"(?<![>\d])>(?!>)\s*\S"),
        Risk.caution,
        "output redirection with > truncates the target file",
        rollback="Overwritten file contents are lost unless backed up.",
        suggestion="Use >> to append, or write to a new file.",
    ),
    RiskRule(
        _rx(r"\bsudo\b"),
        Risk.caution,
        "runs with elevated privileges",
        suggestion="Confirm the command really needs root.",
    ),
    RiskRule(
        _rx(r"\bkill\s+-9\b|\bkillall\b|\bpkill\b"),
        Risk.caution,
        "forcefully terminates processes",
    ),
    RiskRule(
        _rx(r"\b(npm|pnpm|yarn)\s+(install|add|i)\b.*\s(-g|--global)\b|\bpip\s+install\b.*--user"),
        Risk.caution,
        "global package install mutates machine-wide state",
        suggestion="Prefer a project-local install / virtualenv.",
    ),
    RiskRule(
        _rx(r"\b(DROP|TRUNCATE|DELETE)\s+(TABLE|DATABASE|FROM)\b"),
        Risk.destructive,
        "destructive SQL (DROP/TRUNCATE/DELETE)",
        reversible=False,
        rollback="Only recoverable from a database backup.",
        suggestion="Wrap in a transaction and add a WHERE clause / dry-run first.",
    ),
    RiskRule(
        _rx(r"\bdocker\s+(system\s+prune|volume\s+rm|rm\s+-f)"),
        Risk.caution,
        "removes Docker resources/volumes",
        reversible=False,
    ),
]


@dataclass
class StaticAssessment:
    """Result of the static pass."""

    risk: Risk = Risk.safe
    reasons: list[str] = field(default_factory=list)
    reversible: bool = True
    rollback_hint: str | None = None
    suggestions: list[str] = field(default_factory=list)
    # If set, we can answer WITHOUT calling the model.
    short_circuit: CommandPrediction | None = None


class StaticRiskScanner:
    def assess(self, command: str, env: EnvSnapshot | None = None) -> StaticAssessment:
        assessment = self._scan_rules(command)
        short = self._deterministic(command, env, assessment)
        if short is not None:
            assessment.short_circuit = short
        return assessment

    # -- risk rules -----------------------------------------------------------

    def _scan_rules(self, command: str) -> StaticAssessment:
        risk = Risk.safe
        reasons: list[str] = []
        suggestions: list[str] = []
        reversible = True
        rollback: str | None = None

        for rule in RISK_RULES:
            if rule.pattern.search(command):
                risk = max_risk(risk, rule.risk)
                reasons.append(rule.reason)
                if not rule.reversible:
                    reversible = False
                if rule.rollback and rollback is None:
                    rollback = rule.rollback
                if rule.suggestion:
                    suggestions.append(rule.suggestion)

        return StaticAssessment(
            risk=risk,
            reasons=reasons,
            reversible=reversible,
            rollback_hint=rollback,
            suggestions=suggestions,
        )

    # -- deterministic short-circuits ----------------------------------------

    def _deterministic(
        self, command: str, env: EnvSnapshot | None, assessment: StaticAssessment
    ) -> CommandPrediction | None:
        """Predict, without the model, outcomes we already know from the environment."""
        if env is None:
            return None

        first = self._first_token(command)
        if first is None:
            return None

        # 1) First program not installed → command not found (only when we have a tool map
        #    and the token isn't a shell builtin / relative path).
        if (
            env.tools_available  # we actually probed availability
            and first in env.tools_available
            and env.tools_available[first] is False
            and not self._is_builtin(first)
        ):
            shell_name = (env.shell or "sh").rsplit("/", 1)[-1]
            return CommandPrediction(
                command=command,
                predicted_stdout="",
                predicted_stderr=f"{shell_name}: command not found: {first}",
                predicted_exit_code=127,
                risk=assessment.risk,
                risk_reasons=assessment.reasons
                + [f"'{first}' is not installed on this machine"],
                reversible=assessment.reversible,
                rollback_hint=assessment.rollback_hint,
                suggestions=assessment.suggestions
                + [f"Install {first} first, or use an available alternative."],
                confidence=0.97,
                confidence_basis="'%s' is absent from PATH per the environment snapshot." % first,
                assumptions=[],
                source=Source.static,
            )

        return None

    @staticmethod
    def _first_token(command: str) -> str | None:
        stripped = command.strip()
        if not stripped:
            return None
        # Skip leading env-assignments (FOO=bar cmd ...) and `sudo`.
        try:
            tokens = shlex.split(stripped)
        except ValueError:
            tokens = stripped.split()
        for tok in tokens:
            if tok == "sudo":
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):  # VAR=value
                continue
            # A path (./x, /usr/bin/x) is not a bare PATH lookup — don't judge availability.
            if "/" in tok:
                return None
            return tok
        return None

    @staticmethod
    def _is_builtin(name: str) -> bool:
        builtins = {
            "cd", "echo", "export", "pwd", "set", "unset", "alias", "source",
            ".", "test", "true", "false", "read", "printf", "exit", "return",
            "if", "for", "while", "case", "function", "local", "eval", "exec",
            "type", "which", "command", "history", "jobs", "fg", "bg", "wait",
        }
        return name in builtins

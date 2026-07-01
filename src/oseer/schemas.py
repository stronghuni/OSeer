"""Structured request/response schemas returned to the calling agent."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Risk(str, Enum):
    """Severity ordering matters: use :func:`max_risk` to merge assessments."""

    safe = "safe"
    caution = "caution"
    destructive = "destructive"


_RISK_ORDER = {Risk.safe: 0, Risk.caution: 1, Risk.destructive: 2}


def max_risk(*risks: Risk) -> Risk:
    """Return the most severe risk among the arguments (defaults to ``safe``)."""
    if not risks:
        return Risk.safe
    return max(risks, key=lambda r: _RISK_ORDER[r])


class Source(str, Enum):
    """Where a prediction came from, so the agent can weigh it."""

    model = "model"          # world-model prediction (+ static risk merged in)
    static = "static"        # deterministic short-circuit, no model call needed
    degraded = "degraded"    # model/API unavailable; static-only, low confidence


DISCLAIMER = (
    "This is a PREDICTION of what the command would do, not the result of running it. "
    "Treat it as advisory; verify before relying on it for irreversible actions."
)


class EnvSnapshot(BaseModel):
    """Read-only, secret-sanitized facts about the terminal environment."""

    os: str = ""
    os_version: str = ""
    kernel: str = ""
    shell: str = ""
    shell_version: str = ""
    cwd: str = ""
    cwd_listing: list[str] = Field(default_factory=list)
    git: str | None = None  # short status summary if inside a repo
    tools_available: dict[str, bool] = Field(default_factory=dict)
    package_managers: list[str] = Field(default_factory=list)
    env_vars: dict[str, str] = Field(default_factory=dict)  # sanitized; secrets redacted
    grounding: str = "full"
    captured_at: float = 0.0


class CommandPrediction(BaseModel):
    """Predicted outcome of a shell command."""

    command: str
    predicted_stdout: str = ""
    predicted_stderr: str = ""
    predicted_exit_code: int = 0
    filesystem_effects: list[str] = Field(default_factory=list)
    state_changes: list[str] = Field(default_factory=list)

    risk: Risk = Risk.safe
    risk_reasons: list[str] = Field(default_factory=list)
    reversible: bool = True
    rollback_hint: str | None = None
    suggestions: list[str] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_basis: str = ""
    assumptions: list[str] = Field(default_factory=list)

    source: Source = Source.model
    disclaimer: str = DISCLAIMER


class ToolCallPrediction(BaseModel):
    """Predicted outcome of an MCP tool call."""

    server: str
    tool: str
    predicted_result: str = ""
    side_effects: list[str] = Field(default_factory=list)

    risk: Risk = Risk.safe
    risk_reasons: list[str] = Field(default_factory=list)
    reversible: bool = True
    rollback_hint: str | None = None
    suggestions: list[str] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_basis: str = ""
    assumptions: list[str] = Field(default_factory=list)

    source: Source = Source.model
    disclaimer: str = DISCLAIMER

"""OSeer MCP server.

Exposes three tools over MCP (stdio by default):

- ``predict_command``    — dry-run a shell command; returns a predicted result + risk.
- ``predict_tool_call``  — dry-run another MCP tool call.
- ``oseer_env_snapshot`` — show the (sanitized) environment OSeer would send to the model.

Run directly:  ``uv run oseer``   or   ``uv run mcp dev src/oseer/server.py``
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .predict import Predictor
from .schemas import CommandPrediction, EnvSnapshot, ToolCallPrediction

mcp = FastMCP(
    "oseer",
    instructions=(
        "OSeer predicts what a terminal command (or MCP tool call) would do BEFORE you run it. "
        "Call predict_command before executing any command that is destructive, expensive, "
        "long-running, or whose effect is uncertain, then decide based on the predicted result "
        "and risk. Predictions are advisory, not execution."
    ),
)

_predictor = Predictor()


@mcp.tool()
async def predict_command(
    command: str,
    cwd: str | None = None,
    shell: str | None = None,
    reasoning: bool | None = None,
) -> CommandPrediction:
    """Predict the outcome of a shell command WITHOUT running it.

    Call this BEFORE executing any command that is destructive, expensive, long-running,
    or whose effect is uncertain (e.g. rm, git push --force, migrations, bulk edits, or a
    command that might not exist on this machine).

    Returns the predicted stdout / stderr / exit code, filesystem and state changes, a risk
    level (safe | caution | destructive) with reasons, reversibility + rollback hints, safer
    alternative suggestions, and a confidence score. This is a PREDICTION, not execution — use
    it to decide whether and how to run the command.

    Args:
        command: The exact shell command you are considering running.
        cwd: Working directory the command would run in (defaults to OSeer's cwd).
        shell: Target shell (e.g. /bin/zsh); defaults to the detected shell.
        reasoning: Force the model's <think> reasoning on/off for this call (slower, more
            tokens). Omit to use the server default (OSEER_REASONING).
    """
    return await _predictor.predict_command(command, cwd=cwd, shell=shell, reasoning=reasoning)


@mcp.tool()
async def predict_tool_call(
    server: str,
    tool: str,
    arguments: dict,
    context: str | None = None,
) -> ToolCallPrediction:
    """Predict the outcome of another MCP tool call WITHOUT invoking it.

    Use this to dry-run a tool whose effect is uncertain or potentially irreversible (writes,
    deletes, sends, payments). Returns the predicted result, side effects, risk level, rollback
    hints, and confidence.

    Args:
        server: Name of the MCP server the tool belongs to.
        tool: The tool name.
        arguments: The arguments you would pass to the tool.
        context: Optional extra context about the tool's expected behavior.
    """
    return await _predictor.predict_tool_call(server, tool, arguments, context=context)


@mcp.tool()
def oseer_env_snapshot(refresh: bool = False) -> EnvSnapshot:
    """Show the read-only, secret-sanitized environment snapshot OSeer uses to ground predictions.

    Useful for transparency and debugging: it reveals exactly what OSeer knows about this
    machine (OS, shell, cwd contents, git status, available tools, sanitized env vars) and would
    send to the hosted model. Grounding depth is controlled by OSEER_GROUNDING.

    Args:
        refresh: Bypass the TTL cache and re-probe the environment.
    """
    return _predictor.env_snapshot(refresh=refresh)


def main() -> None:
    """Console-script entry point (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()

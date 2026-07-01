---
name: predict
description: >-
  Dry-run a shell command (or MCP tool call) BEFORE executing it to predict its result and risk,
  using the OSeer world model. Use PROACTIVELY before running any command that is destructive,
  expensive, long-running, or uncertain — e.g. rm / rm -rf, git push --force, git reset --hard,
  database migrations or DROP/DELETE, bulk file edits, chmod -R, package installs, curl | sh, or a
  command that might not exist on this machine. Also use when the user asks to "predict",
  "dry-run", "what would this command do", or "is this command safe".
allowed-tools:
  - mcp__plugin_oseer_oseer__predict_command
  - mcp__plugin_oseer_oseer__predict_tool_call
  - mcp__plugin_oseer_oseer__oseer_env_snapshot
  - mcp__oseer__predict_command
  - mcp__oseer__predict_tool_call
  - mcp__oseer__oseer_env_snapshot
---

# OSeer — predict before you run

OSeer predicts what a terminal command would do **without executing it**, grounded in this machine's
real environment (OS, shell, cwd, git state, installed tools), and returns a risk assessment. Use it
as a safety gate so you don't run wrong, wasteful, or irreversible commands.

## When to use (proactively)

Before you would call the Bash tool, first call `predict_command` when the command is:

- **Destructive / irreversible** — `rm`, `rm -rf`, `git push --force`, `git reset --hard`,
  `DROP`/`TRUNCATE`/`DELETE`, `dd`, `mkfs`, `find … -delete`, `chmod -R`, `docker … prune`.
- **Expensive or long-running** — builds, installs, migrations, `docker build`, large downloads.
- **Uncertain** — the effect isn't obvious, the target path may not exist, or the required tool may
  not be installed on this machine.

For routine read-only commands (`ls`, `cat`, `git status`), just run them — no prediction needed.

## How to use it

1. Call `predict_command` with the exact `command` (and `cwd` if not the default).
2. Read the returned prediction:
   - `predicted_stdout` / `predicted_stderr` / `predicted_exit_code`
   - `risk` (`safe` | `caution` | `destructive`), `risk_reasons`, `reversible`, `rollback_hint`
   - `suggestions` (safer/more efficient alternatives), `confidence`, `assumptions`, `source`
3. Decide:
   - **`safe`** → proceed.
   - **`caution`** → proceed carefully; mention the caveat.
   - **`destructive`** → do **not** run it silently. Tell the user what OSeer predicts, especially if
     `reversible` is false, and prefer a `suggestions` alternative or ask the user to confirm.
4. If `source` is `degraded`, the model server was unreachable — you still get the rule-based risk;
   treat low-confidence predictions accordingly.

## Predicting MCP tool calls

Before invoking another MCP tool whose effect is uncertain or irreversible (writes, deletes, sends,
payments), call `predict_tool_call` with the `server`, `tool`, and `arguments` to
predict its result and side effects first.

## Notes

- Predictions are **advisory**, not execution, and the model is imperfect — always weigh `confidence`
  and `assumptions`, and never treat a prediction as the real result for irreversible actions.
- `oseer_env_snapshot` shows exactly what OSeer knows about this machine (useful for
  debugging why a prediction looks off).

# OSeer — OS Seer 🔮

<p align="center">
  <img src="public/oracle_crystal_ball.png" alt="OSeer — the OS oracle" width="360">
</p>

**An MCP server that lets an AI agent dry-run a terminal command _before_ executing it.**

Coding agents run commands blind — they execute, then react to failures, wasted work, or
irreversible damage. OSeer ("OS Seer") predicts what a command *would* do — **stdout, stderr, exit
code, filesystem effects** — grounded in your real machine, and flags destructive or wrong commands
so the agent decides *before* running. The predictor is a purpose-built world model,
[**Qwen/Qwen-AgentWorld-35B-A3B**](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B), served on
your own OpenAI-compatible endpoint (vLLM / SGLang / Ollama).

> ⚠️ Predictions are **advisory, not execution**. The model is probabilistic, so every prediction
> carries a confidence score and stated assumptions — and a **rule-based safety check** backs it up
> even when the model is wrong or offline. Never treat a prediction as ground truth for irreversible
> actions.

---

## Features

- 🔮 **Predicts before running** — stdout, stderr, exit code, filesystem & state changes for any
  shell command, without executing it.
- 🛡️ **Risk gate** — classifies every command `safe` / `caution` / `destructive`, with reasons,
  reversibility, rollback hints, and safer alternatives. A rule-based floor catches `rm -rf`,
  `git push --force`, `DROP TABLE`, `curl | sh`, etc. — even if the model underrates them.
- 🌍 **Grounded in *your* machine** — reads (read-only) your OS, shell, cwd, git state, and which
  tools are installed, so predictions match your environment, not a generic Linux box.
- ⚡ **Fast & cheap where it can be** — a missing tool or bad path is answered instantly with **no
  model call**; identical dry-runs are cached.
- 🔌 **Predicts MCP tool calls too** — dry-run another tool (writes, deletes, payments) before
  invoking it.
- 🔒 **Private & safe** — self-hosted model (nothing leaves your infra), secrets stripped from the
  environment snapshot, and OSeer **never executes** the command it predicts.
- 🧩 **Ships as a Claude Code plugin** — one install adds both the MCP server and a skill that makes
  the agent use it automatically.

---

## Skill & tools

Installing OSeer gives your agent one **skill** and three **MCP tools**.

### Skill: `/oseer:predict`

Auto-invokes before destructive/uncertain commands (or call it manually). It tells the agent to
predict first, read the risk, and — for `destructive` results — surface the prediction and prefer a
safer alternative instead of running silently. See [`skills/predict/SKILL.md`](skills/predict/SKILL.md).

### Tools

| Tool | What it does |
|------|--------------|
| `predict_command(command, cwd?, shell?, reasoning?)` | Predict a shell command's stdout / stderr / exit code, filesystem & state changes, risk, rollback hints, safer alternatives, confidence. |
| `predict_tool_call(server, tool, arguments, context?)` | Predict the result and side effects of another MCP tool call. |
| `oseer_env_snapshot(refresh?)` | Show the sanitized environment OSeer would send to the model. |

---

## Install & use

**1. Serve the model** on any OpenAI-compatible server (no API key needed):

```bash
vllm serve Qwen/Qwen-AgentWorld-35B-A3B --served-model-name Qwen/Qwen-AgentWorld-35B-A3B --port 8000
```

**2. Install the plugin** in Claude Code (export your endpoint first so the plugin's server sees it):

```bash
export OSEER_API_BASE=http://your-host:8000/v1
export OSEER_MODEL=Qwen/Qwen-AgentWorld-35B-A3B

# in Claude Code:
/plugin marketplace add stronghuni/OSeer
/plugin install oseer@oseer
```

**3. Use it.** The agent now dry-runs risky commands on its own. In practice:

```
You:   Clean up the repo — drop node_modules and dist.

Agent: (calls predict_command "rm -rf node_modules dist")
       OSeer says this is DESTRUCTIVE and irreversible, but here node_modules/ and dist/
       don't exist, so it would be a harmless no-op (exit 1). Want me to run it anyway,
       or scope it to what's actually there?
```

You can also trigger it explicitly with `/oseer:predict <command>`.

> Prefer not to install the full plugin? Register just the MCP server:
> ```bash
> claude mcp add oseer -e OSEER_API_BASE=http://your-host:8000/v1 \
>   -e OSEER_MODEL=Qwen/Qwen-AgentWorld-35B-A3B -- uv run --directory /path/to/OSeer oseer
> ```

---

## Expected call results

Real outputs from the live model (fields trimmed for readability).

**Safe command** — `predict_command("ls -la")`:
```json
{ "risk": "safe", "predicted_exit_code": 0, "confidence": 1.0, "source": "model" }
```

**Destructive command** — `predict_command("git push --force origin main")`:
```json
{
  "risk": "destructive",
  "reversible": false,
  "risk_reasons": ["force push can overwrite remote history for others"],
  "rollback_hint": "Recoverable only via reflog if someone still has the old commits.",
  "suggestions": ["Use --force-with-lease, or push to a new branch."],
  "confidence": 0.95,
  "source": "model"
}
```

**Missing tool** — `predict_command("git status")` on a machine without git (answered instantly, **no model call**):
```json
{
  "predicted_stderr": "bash: command not found: git",
  "predicted_exit_code": 127,
  "confidence": 0.97,
  "source": "static"
}
```

**MCP tool call** — `predict_tool_call("database", "drop_table", {"table": "users"})`:
```json
{
  "risk": "destructive",
  "reversible": false,
  "side_effects": ["The 'users' table and all its data are permanently removed from the database."],
  "rollback_hint": "Recover from a database backup, or recreate and repopulate the table.",
  "confidence": 0.95,
  "source": "model"
}
```

**Environment snapshot** — `oseer_env_snapshot()` (what OSeer knows about this machine; secrets redacted):
```json
{
  "os": "macOS", "os_version": "26.5.1", "shell": "/bin/zsh",
  "cwd": "/Users/you/project",
  "git": "branch=main clean origin=https://github.com/you/project.git ahead=0 behind=0",
  "package_managers": ["brew", "npm", "uv"],
  "tools_available": { "git": true, "docker": true, "node": true }
}
```

Every prediction also includes an `assumptions` list and a `disclaimer` making clear it is a
prediction, not a real result.

---

## Configuration

All settings are environment variables (prefix `OSEER_`). Highlights:

| Var | Default | Meaning |
|-----|---------|---------|
| `OSEER_API_BASE` | `http://localhost:8000/v1` | Your OpenAI-compatible endpoint (vLLM :8000 · SGLang :30000 · Ollama :11434) |
| `OSEER_MODEL` | `Qwen/Qwen-AgentWorld-35B-A3B` | Model id as your server exposes it |
| `OSEER_API_KEY` | — | Optional; leave blank for a keyless self-hosted server |
| `OSEER_GROUNDING` | `full` | `full` \| `minimal` \| `none` — how much environment to send |
| `OSEER_REASONING` | `false` | Model's `<think>` mode (slower, higher quality); per-call override on the tool |
| `OSEER_JSON_MODE` | `true` | Force valid JSON via `response_format` (disable for stricter servers) |
| `OSEER_TIMEOUT` / `OSEER_RETRIES` | `60` / `2` | Request timeout & retry attempts |

See [`.env.example`](.env.example) for the full list.

## Privacy

With a self-hosted server the command and environment snapshot **never leave your infrastructure**.
OSeer also strips secrets (any env var named like `TOKEN|KEY|SECRET|PASSWORD|…` is redacted before
sending), lets you reduce what's shared (`OSEER_GROUNDING=minimal` or `none`), and never executes
the command it predicts.

## License

MIT

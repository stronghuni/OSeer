# OSeer — OS Seer 🔮

<p align="center">
  <img src="public/oracle_crystal_ball.png" alt="OSeer — the OS oracle" width="360">
</p>

**An MCP server that lets an AI agent dry-run a terminal command _before_ executing it.**

Coding agents run commands blind: they execute, then react to failures, wasted work, or
irreversible damage (`rm -rf`, `git push --force`, a command that needs a tool the machine doesn't
have). OSeer ("OS Seer") predicts what a command would do — **stdout, stderr, exit code, filesystem
effects** — and flags destructive or wrong commands, so the agent can decide *before* running.

The predictor is [**Qwen/Qwen-AgentWorld-35B-A3B**](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B),
a purpose-built "language world model" trained to predict the next environment state given an agent
action, across Terminal / OS / MCP / SWE domains. OSeer talks to it via any **OpenAI-compatible**
endpoint — designed for a **self-hosted** vLLM / SGLang / Ollama server (no API key required).

> ⚠️ **Predictions are advisory, not execution.** The model is probabilistic and imperfect. OSeer
> always returns a **confidence score and stated assumptions**, and backs every prediction with a
> model-independent static safety check. Never treat a prediction as ground truth for irreversible
> actions.

---

## How it works

```
Agent ──MCP──▶ OSeer
                 1. EnvironmentProbe   read-only snapshot of THIS machine (OS, shell, cwd,
                                       git, installed tools, sanitized env vars)
                 2. StaticRiskScanner  rule-based risk + deterministic short-circuits
                                       (missing tool → "command not found", no model call)
                 3. World-model prompt  grounded in the real environment
                 4. your server ──▶  Qwen-AgentWorld-35B-A3B   (vLLM / SGLang / Ollama)
                 5. Parse + merge      model prediction + static risk floor (max severity)
               ◀── CommandPrediction   stdout/stderr/exit code, risk, rollback, confidence
```

Key guarantees:
- **OSeer never executes** the command being predicted — every probe is read-only.
- **Static safety floor:** destructive commands stay flagged even if the model is unreachable or
  wrong about risk. If the API fails, OSeer degrades to a static-only prediction — never an error.
- **Cost/latency savers:** identical dry-runs are cached; deterministic outcomes (e.g. a missing
  tool) skip the model entirely.

## Tools

| Tool | What it does |
|------|--------------|
| `predict_command(command, cwd?, shell?, reasoning?)` | Predict a shell command's stdout / stderr / exit code, filesystem & state changes, risk level, rollback hints, safer alternatives, and confidence. |
| `predict_tool_call(server, tool, arguments, context?)` | Predict the result and side effects of another MCP tool call. |
| `oseer_env_snapshot(refresh?)` | Show the sanitized environment OSeer would send to the model (transparency/debug). |

**Example** — `predict_command("git push --force origin main")` returns:

```json
{
  "command": "git push --force origin main",
  "predicted_exit_code": 0,
  "risk": "destructive",
  "risk_reasons": ["force push can overwrite remote history for others"],
  "reversible": false,
  "rollback_hint": "Recoverable only via reflog if someone still has the old commits.",
  "suggestions": ["Use --force-with-lease, or push to a new branch."],
  "confidence": 0.95,
  "source": "model"
}
```

## Setup

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).

**1. Serve the model** with any OpenAI-compatible server, e.g. vLLM:

```bash
vllm serve Qwen/Qwen-AgentWorld-35B-A3B \
  --served-model-name Qwen/Qwen-AgentWorld-35B-A3B \
  --port 8000
# → OpenAI-compatible API at http://localhost:8000/v1  (no key needed)
```

**2. Configure OSeer** to point at it:

```bash
uv sync                 # install dependencies
cp .env.example .env    # then set OSEER_API_BASE / OSEER_MODEL to match your server
```

```bash
OSEER_API_BASE=http://localhost:8000/v1     # vLLM :8000 · SGLang :30000 · Ollama :11434
OSEER_MODEL=Qwen/Qwen-AgentWorld-35B-A3B    # exactly as your server exposes it
OSEER_API_KEY=                              # leave blank for a keyless self-hosted server
```

The endpoint is any OpenAI-compatible Chat Completions host, so a dedicated/hosted endpoint works
too — just set the base URL, model/endpoint id, and (if required) a key. See
[`.env.example`](.env.example) for all options.

**Server compatibility:** against vLLM/SGLang, OSeer forces valid JSON (`response_format`), toggles
`<think>` via `chat_template_kwargs.enable_thinking`, and sends `top_k`. If your server rejects any
of these, disable them with `OSEER_JSON_MODE=false`, `OSEER_SEND_THINKING_FLAG=false`, or
`OSEER_SEND_TOP_K=false`.

### Install as a Claude Code plugin (MCP server **+** skill)

OSeer ships as a Claude Code plugin. Installing it gives you **both** the MCP server and a
`predict` **skill** that makes the agent reach for OSeer proactively before risky commands.

First export your model endpoint (inherited by the plugin's server), then install:

```bash
export OSEER_API_BASE=http://your-host:8000/v1     # your self-hosted server
export OSEER_MODEL=Qwen/Qwen-AgentWorld-35B-A3B

# In Claude Code:
/plugin marketplace add stronghuni/OSeer
/plugin install oseer@oseer
```

That's it. Now:

- The **`oseer` MCP server** exposes `predict_command`, `predict_tool_call`, `oseer_env_snapshot`
  (as `mcp__plugin_oseer_oseer__*` when installed as a plugin, or `mcp__oseer__*` if you add the
  server directly).
- The **`predict` skill** auto-invokes before destructive/uncertain commands, and you can call it
  manually with `/oseer:predict`. See [`skills/predict/SKILL.md`](skills/predict/SKILL.md).

### Or register just the MCP server

```bash
claude mcp add oseer \
  -e OSEER_API_BASE=http://your-host:8000/v1 \
  -e OSEER_MODEL=Qwen/Qwen-AgentWorld-35B-A3B \
  -- uv run --directory /path/to/OSeer oseer
```

### Explore interactively

```bash
uv run mcp dev src/oseer/server.py                                            # MCP Inspector
OSEER_API_BASE=http://your-host:8000/v1 uv run python scripts/smoke.py        # 4-command smoke test
OSEER_API_BASE=http://your-host:8000/v1 uv run python scripts/scenarios.py    # 5-persona matrix
OSEER_API_BASE=http://your-host:8000/v1 uv run python scripts/verify_live.py  # 12-check verification
```

## Configuration

All settings are environment variables (prefix `OSEER_`). Highlights:

| Var | Default | Meaning |
|-----|---------|---------|
| `OSEER_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible endpoint (your server) |
| `OSEER_API_KEY` | — | Bearer token; optional for self-hosted (`FRIENDLI_TOKEN` also accepted) |
| `OSEER_MODEL` | `Qwen/Qwen-AgentWorld-35B-A3B` | Model / endpoint id |
| `OSEER_GROUNDING` | `full` | `full` \| `minimal` \| `none` — how much environment to send |
| `OSEER_REASONING` | `false` | Default for the model's `<think>` mode (slower); per-call override on the tool |
| `OSEER_REASONING_MAX_TOKENS` | `8192` | `max_tokens` when reasoning (room for the think trace) |
| `OSEER_JSON_MODE` | `true` | Force valid JSON via `response_format` when not reasoning |
| `OSEER_SEND_TOP_K` / `OSEER_SEND_THINKING_FLAG` | `true` | Vendor extensions for vLLM/SGLang |
| `OSEER_TIMEOUT` / `OSEER_RETRIES` | `60` / `2` | API timeout & retry attempts |
| `OSEER_ENV_TTL` | `60` | Environment snapshot cache TTL (s) |

## Privacy

With a **self-hosted** server, the command and environment snapshot never leave your infrastructure.
OSeer still defends in depth so it's safe against any endpoint:

- **Secrets are stripped** — any env var whose name matches `TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH|…`
  has its value redacted before anything is sent.
- **Grounding is configurable** — `OSEER_GROUNDING=minimal` drops cwd contents and env vars;
  `none` sends no environment at all.
- Use `oseer_env_snapshot` to see exactly what would be sent.

## Development

```bash
uv sync --extra dev
uv run pytest              # full offline suite (no API key needed)
```

The **88 offline unit tests** are fully deterministic (mocked provider + stub environment — no server
needed). Three scripts exercise a **real** server: `scripts/smoke.py` (4 commands),
`scripts/scenarios.py` (the same commands across 5 synthetic user personas — macOS/zsh, Ubuntu/bash,
Alpine, Python venv, no-git — asserting the safety invariants hold in every environment), and
`scripts/verify_live.py` (a 12-check end-to-end harness: grounding, secret sanitization, all risk
tiers, static short-circuit, reasoning, caching, and graceful degradation). Layout:

```
.claude-plugin/
  plugin.json      Claude Code plugin manifest (bundles the MCP server, inline)
  marketplace.json marketplace entry for `/plugin marketplace add`
skills/predict/
  SKILL.md         agent skill that invokes the MCP tools proactively
src/oseer/
  server.py        FastMCP server + tool definitions
  predict.py       orchestrator (env → static → model → parse → merge)
  environment.py   read-only, cached, secret-sanitized environment probe
  safety.py        static risk scanner + deterministic short-circuits
  providers/       OpenAI-compatible model client
  prompts/         terminal + MCP domain world-model prompts (env/path/tool grounding)
  parsing.py       tolerant JSON extraction + field coercion
  schemas.py       Pydantic prediction schemas
scripts/           smoke · scenarios · verify_live (live-server checks)
tests/             offline pytest suite
```

## License

MIT

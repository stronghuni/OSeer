# OSeer — OS Seer 🔮

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

### Register with Claude Code

```bash
claude mcp add oseer -- uv run --directory /Users/namuneulbo/Desktop/OSeer oseer
```

Or add to `.mcp.json`:

```json
{
  "mcpServers": {
    "oseer": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/namuneulbo/Desktop/OSeer", "oseer"],
      "env": { "OSEER_API_KEY": "your_friendli_token" }
    }
  }
}
```

### Explore interactively

```bash
uv run mcp dev src/oseer/server.py    # MCP Inspector
uv run python scripts/smoke.py        # live smoke test (needs an API key)
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

Tests are fully offline (mocked provider + stub environment). Layout:

```
src/oseer/
  server.py       FastMCP server + tool definitions
  predict.py      orchestrator (env → static → model → parse → merge)
  environment.py  read-only, cached, secret-sanitized environment probe
  safety.py       static risk scanner + deterministic short-circuits
  providers/      OpenAI-compatible model client (FriendliAI)
  prompts/        terminal + MCP domain world-model prompts
  parsing.py      tolerant JSON extraction + field coercion
  schemas.py      Pydantic prediction schemas
```

## License

MIT

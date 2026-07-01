"""Configuration for OSeer, loaded from environment variables (and an optional .env).

All settings are prefixed with ``OSEER_`` except the model credentials, which also
accept the conventional ``FRIENDLI_TOKEN`` / ``OSEER_API_KEY`` names.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen-AgentWorld-35B-A3B"


class Grounding(str, Enum):
    """How much of the real terminal environment to send to the model.

    - ``full``:    OS, shell, cwd listing, git status, tool availability, sanitized env vars.
    - ``minimal``: OS, shell, and tool availability only (no cwd contents / env vars).
    - ``none``:    send no environment context at all (maximum privacy).
    """

    full = "full"
    minimal = "minimal"
    none = "none"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OSEER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model API (any OpenAI-compatible Chat Completions host: self-hosted vLLM/SGLang,
    #     Ollama, FriendliAI, …). Default targets a locally self-served vLLM endpoint. ---
    api_base: str = Field(
        default=DEFAULT_API_BASE,
        description="OpenAI-compatible base URL (e.g. vLLM :8000/v1, SGLang :30000/v1, Ollama :11434/v1).",
    )
    api_key: str = Field(
        default="",
        description="Bearer token. OPTIONAL for self-hosted servers (env: OSEER_API_KEY / FRIENDLI_TOKEN).",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model id as your server exposes it (or a dedicated endpoint id).",
    )

    @field_validator("api_base", "model", mode="before")
    @classmethod
    def _blank_to_default(cls, v, info):
        # An unset ${OSEER_API_BASE} passthrough arrives as "" — fall back to the default
        # rather than overriding it with an empty string.
        if v is None or (isinstance(v, str) and not v.strip()):
            return DEFAULT_API_BASE if info.field_name == "api_base" else DEFAULT_MODEL
        return v

    # --- Sampling (defaults from the model card) ---
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 2048
    reasoning: bool = Field(
        default=False,
        description="Default for the model's <think> reasoning mode (slower, more tokens).",
    )
    reasoning_max_tokens: int = Field(
        default=8192,
        description="max_tokens used when reasoning is on (needs room for the think trace + JSON).",
    )

    # --- Server-compatibility toggles (all safe for vLLM/SGLang; disable for stricter hosts) ---
    json_mode: bool = Field(
        default=True,
        description="Send response_format={'type':'json_object'} when NOT reasoning, to force valid JSON.",
    )
    send_top_k: bool = Field(
        default=True,
        description="Include top_k in the payload (a vendor extension; vLLM/SGLang accept it).",
    )
    send_thinking_flag: bool = Field(
        default=True,
        description="Send chat_template_kwargs={'enable_thinking': ...} to control <think> per request.",
    )

    # --- Behavior ---
    timeout: float = Field(default=60.0, description="Per-request timeout in seconds.")
    retries: int = Field(default=2, description="Retry attempts on transient API errors.")
    env_ttl: float = Field(default=60.0, description="Environment snapshot cache TTL (seconds).")
    grounding: Grounding = Grounding.full
    cache_size: int = Field(default=256, description="Max entries in the prediction cache.")

    def api_key_present(self) -> bool:
        return bool(self.api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings.

    The extra ``FRIENDLI_TOKEN`` fallback keeps parity with FriendliAI's own tooling.
    """
    import os

    settings = Settings()
    if not settings.api_key_present():
        token = os.environ.get("FRIENDLI_TOKEN", "")
        if token:
            settings = settings.model_copy(update={"api_key": token})
    return settings

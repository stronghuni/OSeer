"""Model provider protocol.

OSeer talks to the world model through a tiny protocol so any OpenAI-compatible host
(FriendliAI serverless, a dedicated endpoint, vLLM, SGLang, …) can be swapped in via
configuration alone. Only :class:`~oseer.providers.friendli.FriendliProvider` is
implemented for v1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Raised when the model API cannot produce a completion (network, auth, 5xx…)."""


Message = dict[str, str]  # {"role": ..., "content": ...}


@runtime_checkable
class ModelProvider(Protocol):
    async def complete(self, messages: list[Message], reasoning: bool = False) -> str:
        """Return the raw assistant text for a chat completion, or raise ProviderError.

        ``reasoning`` toggles the model's <think> mode for this request.
        """
        ...

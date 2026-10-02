"""The provider protocol.

A provider turns a normalized :class:`Request` into one model's answer. It knows nothing
about routes, tiers, prices, or why it was chosen; the router owns all of that.

Every failure leaves a provider as a :class:`~dumbwaiter.errors.ProviderError` (or a
subclass). Escalation policy branches on those types, so a raw httpx, botocore, or
anthropic exception escaping a provider is a bug.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dumbwaiter.types import Completion, Request


@runtime_checkable
class Provider(Protocol):
    name: str
    """The prefix used in model specs, e.g. ``ollama`` in ``ollama/llama3.1:latest``."""

    async def complete(self, request: Request, model_id: str) -> Completion:
        """Run one non-streaming completion against ``model_id``."""
        ...

    async def aclose(self) -> None:
        """Release connections. Safe to call more than once."""
        ...

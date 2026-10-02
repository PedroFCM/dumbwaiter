"""Providers: one per backend, all behind :class:`Provider`."""

from __future__ import annotations

from dumbwaiter.providers.base import Provider
from dumbwaiter.providers.ollama import OllamaProvider
from dumbwaiter.providers.registry import ProviderFactory, ProviderRegistry

__all__ = ["OllamaProvider", "Provider", "ProviderFactory", "ProviderRegistry"]

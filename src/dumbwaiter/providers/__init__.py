"""Providers: one per backend, all behind :class:`Provider`."""

from __future__ import annotations

from dumbwaiter.providers.anthropic import AnthropicProvider
from dumbwaiter.providers.base import Provider
from dumbwaiter.providers.bedrock import BedrockProvider
from dumbwaiter.providers.ollama import OllamaProvider
from dumbwaiter.providers.registry import ProviderFactory, ProviderRegistry

__all__ = [
    "AnthropicProvider",
    "BedrockProvider",
    "OllamaProvider",
    "Provider",
    "ProviderFactory",
    "ProviderRegistry",
]

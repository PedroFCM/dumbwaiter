"""Name -> provider lookup.

Providers are built on first use, not at registration, so a config that never routes to
Bedrock never imports boto3 and never needs AWS credentials.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from dumbwaiter.errors import ConfigError
from dumbwaiter.providers.base import Provider
from dumbwaiter.providers.ollama import OllamaProvider

ProviderFactory = Callable[[], Provider]


def default_factories() -> dict[str, ProviderFactory]:
    return {"ollama": OllamaProvider}


class ProviderRegistry:
    def __init__(self, factories: Mapping[str, ProviderFactory] | None = None) -> None:
        self._factories = dict(default_factories() if factories is None else factories)
        self._instances: dict[str, Provider] = {}

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._factories)

    def register(self, name: str, factory: ProviderFactory) -> None:
        """Add or replace a provider. Replacing drops any instance already built."""
        self._factories[name] = factory
        self._instances.pop(name, None)

    def require(self, names: Iterable[str]) -> None:
        """Fail now, at load, if any of ``names`` has no provider."""
        unknown = sorted(set(names) - self.names)
        if unknown:
            raise ConfigError(f"unknown provider(s) {unknown}; have {sorted(self.names)}")

    def get(self, name: str) -> Provider:
        if name not in self._instances:
            self.require([name])
            self._instances[name] = self._factories[name]()
        return self._instances[name]

    async def aclose(self) -> None:
        """Close every provider that was built. Ones never used were never opened."""
        instances = list(self._instances.values())
        self._instances.clear()
        for provider in instances:
            await provider.aclose()

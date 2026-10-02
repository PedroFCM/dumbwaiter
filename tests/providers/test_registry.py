from __future__ import annotations

import pytest

from dumbwaiter.errors import ConfigError
from dumbwaiter.providers import OllamaProvider, ProviderRegistry
from dumbwaiter.types import Completion, Request, Usage


class FakeProvider:
    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.closed = 0

    async def complete(self, request: Request, model_id: str) -> Completion:
        return Completion(text=f"{model_id}: {request.routing_text}", usage=Usage())

    async def aclose(self) -> None:
        self.closed += 1


class TestDefaults:
    def test_ollama_is_available_without_any_extra(self):
        assert "ollama" in ProviderRegistry().names

    def test_builds_ollama_on_first_use(self):
        assert isinstance(ProviderRegistry().get("ollama"), OllamaProvider)


class TestLookup:
    def test_instances_are_built_once_and_reused(self):
        built: list[FakeProvider] = []

        def factory() -> FakeProvider:
            built.append(FakeProvider())
            return built[-1]

        registry = ProviderRegistry({"fake": factory})
        assert registry.get("fake") is registry.get("fake")
        assert len(built) == 1

    def test_factories_are_not_called_until_needed(self):
        # The point of laziness: an unused cloud provider never imports its SDK.
        def explode() -> FakeProvider:
            raise AssertionError("built eagerly")

        registry = ProviderRegistry({"fake": explode})
        assert registry.names == {"fake"}

    def test_unknown_provider_names_what_exists(self):
        registry = ProviderRegistry({"fake": FakeProvider})
        with pytest.raises(ConfigError, match=r"unknown provider\(s\) \['nope'\]; have \['fake'\]"):
            registry.get("nope")

    def test_require_reports_every_unknown_name(self):
        registry = ProviderRegistry({"fake": FakeProvider})
        with pytest.raises(ConfigError, match=r"\['a', 'b'\]"):
            registry.require(["b", "fake", "a"])

    def test_require_passes_for_known_names(self):
        ProviderRegistry({"fake": FakeProvider}).require(["fake"])

    def test_register_replaces_and_drops_the_old_instance(self):
        registry = ProviderRegistry({"fake": FakeProvider})
        old = registry.get("fake")
        registry.register("fake", lambda: FakeProvider("fake"))
        assert registry.get("fake") is not old


class TestClose:
    async def test_closes_only_built_providers_and_forgets_them(self):
        fake = FakeProvider()
        registry = ProviderRegistry({"fake": lambda: fake, "unused": FakeProvider})
        registry.get("fake")
        await registry.aclose()
        await registry.aclose()
        assert fake.closed == 1

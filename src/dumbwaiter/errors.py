"""Exception hierarchy.

Everything raised by this library derives from :class:`DumbwaiterError`, so a caller
can wrap the whole thing in one ``except`` without swallowing unrelated failures.
"""

from __future__ import annotations


class DumbwaiterError(Exception):
    """Base class for every error raised by dumbwaiter."""


class ConfigError(DumbwaiterError):
    """Configuration is malformed, inconsistent, or references something unknown."""


class ClassifierError(DumbwaiterError):
    """A classifier could not produce a prediction."""


class RoutingError(DumbwaiterError):
    """A request could not be routed."""


class NoRouteError(RoutingError):
    """No route matched and no usable default was available."""


class BudgetExceeded(RoutingError):
    """Every candidate route would cost more than the configured per-request cap."""

    def __init__(self, estimated_usd: float, cap_usd: float) -> None:
        self.estimated_usd = estimated_usd
        self.cap_usd = cap_usd
        super().__init__(f"estimated ${estimated_usd:.4f} exceeds cap ${cap_usd:.4f}")


class ProviderError(DumbwaiterError):
    """A provider call failed.

    Carries the provider and model so escalation policy can decide whether to retry
    one tier up rather than surfacing the failure.
    """

    def __init__(self, message: str, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(f"[{provider}/{model}] {message}")


class ProviderTimeout(ProviderError):
    """A provider call exceeded its deadline."""


class RateLimited(ProviderError):
    """A provider rejected the call for rate-limit reasons."""

    def __init__(
        self, message: str, *, provider: str, model: str, retry_after: float | None = None
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, provider=provider, model=model)

"""The exception hierarchy is load-bearing for escalation policy.

M4 decides whether to retry one tier up by looking at *which* exception came back and
what it carries, so the shape of these classes is behaviour, not decoration.
"""

from __future__ import annotations

import pytest

from dumbwaiter.errors import (
    BudgetExceeded,
    ClassifierError,
    ConfigError,
    DumbwaiterError,
    NoRouteError,
    ProviderError,
    ProviderTimeout,
    RateLimited,
    RoutingError,
)

ALL_ERRORS = [
    ConfigError,
    ClassifierError,
    RoutingError,
    NoRouteError,
    BudgetExceeded,
    ProviderError,
    ProviderTimeout,
    RateLimited,
]


class TestHierarchy:
    @pytest.mark.parametrize("error_type", ALL_ERRORS)
    def test_everything_derives_from_the_root(self, error_type: type[Exception]):
        # One `except DumbwaiterError` must catch the whole library.
        assert issubclass(error_type, DumbwaiterError)

    @pytest.mark.parametrize("error_type", [NoRouteError, BudgetExceeded])
    def test_routing_failures_share_a_base(self, error_type: type[Exception]):
        assert issubclass(error_type, RoutingError)

    @pytest.mark.parametrize("error_type", [ProviderTimeout, RateLimited])
    def test_provider_failures_share_a_base(self, error_type: type[Exception]):
        # Escalation catches ProviderError; the subclasses must not escape it.
        assert issubclass(error_type, ProviderError)

    def test_config_errors_are_not_routing_errors(self):
        # A bad config is a startup bug, not something to escalate around at runtime.
        assert not issubclass(ConfigError, RoutingError)


class TestProviderError:
    def test_message_identifies_the_failing_model(self):
        error = ProviderError("connection reset", provider="bedrock", model="amazon.nova-lite-v1:0")
        assert str(error) == "[bedrock/amazon.nova-lite-v1:0] connection reset"

    def test_carries_provider_and_model_for_escalation(self):
        error = ProviderError("boom", provider="anthropic", model="claude-opus-5")
        assert error.provider == "anthropic"
        assert error.model == "claude-opus-5"

    def test_timeout_keeps_the_same_shape(self):
        error = ProviderTimeout("deadline exceeded", provider="bedrock", model="m")
        assert error.provider == "bedrock"
        assert "deadline exceeded" in str(error)

    def test_is_raisable_and_catchable_as_its_base(self):
        with pytest.raises(ProviderError):
            raise ProviderTimeout("slow", provider="bedrock", model="m")


class TestRateLimited:
    def test_retry_after_defaults_to_none(self):
        error = RateLimited("slow down", provider="anthropic", model="claude-opus-5")
        assert error.retry_after is None

    def test_retry_after_is_preserved(self):
        # A backoff loop reads this instead of guessing an interval.
        error = RateLimited(
            "slow down", provider="anthropic", model="claude-opus-5", retry_after=30.0
        )
        assert error.retry_after == 30.0

    def test_still_formats_like_a_provider_error(self):
        error = RateLimited("429", provider="bedrock", model="amazon.nova-lite-v1:0")
        assert str(error) == "[bedrock/amazon.nova-lite-v1:0] 429"


class TestBudgetExceeded:
    def test_message_reports_both_numbers(self):
        error = BudgetExceeded(estimated_usd=0.12, cap_usd=0.05)
        assert "0.1200" in str(error)
        assert "0.0500" in str(error)

    def test_carries_the_numbers_for_callers(self):
        error = BudgetExceeded(estimated_usd=0.12, cap_usd=0.05)
        assert error.estimated_usd == pytest.approx(0.12)
        assert error.cap_usd == pytest.approx(0.05)

    def test_is_a_routing_failure_not_a_provider_failure(self):
        # Nothing was called, so escalation must not treat it as a retryable outage.
        error = BudgetExceeded(estimated_usd=1.0, cap_usd=0.1)
        assert isinstance(error, RoutingError)
        assert not isinstance(error, ProviderError)

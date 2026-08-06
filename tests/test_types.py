from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from dumbwaiter.errors import ConfigError
from dumbwaiter.types import Message, ModelRef, Prediction, Request, TokenPrices, Usage


class TestModelRef:
    def test_parses_provider_and_model(self):
        ref = ModelRef.parse("anthropic/claude-sonnet-5")
        assert ref.provider == "anthropic"
        assert ref.model_id == "claude-sonnet-5"

    def test_splits_on_first_separator_only(self):
        # Bedrock ids carry dots and colons; only the first slash is the separator.
        ref = ModelRef.parse("bedrock/amazon.nova-lite-v1:0")
        assert ref.model_id == "amazon.nova-lite-v1:0"

    def test_round_trips_through_str(self):
        spec = "bedrock/anthropic.claude-sonnet-5"
        assert str(ModelRef.parse(spec)) == spec

    @pytest.mark.parametrize("spec", ["claude-sonnet-5", "/claude-sonnet-5", "anthropic/", ""])
    def test_rejects_malformed_specs(self, spec: str):
        with pytest.raises(ConfigError, match="must be"):
            ModelRef.parse(spec)


class TestRequest:
    def test_user_shorthand_builds_single_turn(self):
        req = Request.user("hello")
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    def test_routing_text_is_the_last_user_turn(self):
        req = Request(
            messages=[
                Message(role="user", content="first question"),
                Message(role="assistant", content="an answer"),
                Message(role="user", content="the one that matters"),
            ]
        )
        assert req.routing_text == "the one that matters"

    def test_routing_text_is_empty_without_a_user_turn(self):
        req = Request(messages=[Message(role="assistant", content="orphan")])
        assert req.routing_text == ""

    def test_rejects_empty_messages(self):
        with pytest.raises(ValueError, match="must not be empty"):
            Request(messages=[])

    def test_rejects_non_positive_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            Request.user("hi", max_tokens=0)


class TestUsageAndPricing:
    def test_total_tokens_sums_both_directions(self):
        assert Usage(input_tokens=100, output_tokens=50).total_tokens == 150

    def test_cost_is_exact_decimal_arithmetic(self):
        prices = TokenPrices(
            input_per_mtok=Decimal("3.00"),
            output_per_mtok=Decimal("15.00"),
            source="test",
            checked_on=date(2026, 8, 6),
        )
        # 1M in + 1M out at $3/$15.
        cost = prices.cost(Usage(input_tokens=1_000_000, output_tokens=1_000_000))
        assert cost == Decimal("18")

    def test_cost_of_nothing_is_zero(self):
        prices = TokenPrices(
            input_per_mtok=Decimal("1"),
            output_per_mtok=Decimal("5"),
            source="test",
            checked_on=date(2026, 8, 6),
        )
        assert prices.cost(Usage()) == Decimal(0)


class TestPrediction:
    def test_runner_up_is_the_second_best_label(self):
        pred = Prediction(
            label="complex",
            scores={"simple": 0.1, "code": 0.3, "complex": 0.6},
            confidence=0.3,
            classifier="embedding",
        )
        assert pred.runner_up == "code"

    def test_runner_up_is_none_with_a_single_score(self):
        pred = Prediction(
            label="simple", scores={"simple": 1.0}, confidence=1.0, classifier="rules"
        )
        assert pred.runner_up is None

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_rejects_out_of_range_confidence(self, confidence: float):
        with pytest.raises(ValueError, match="confidence"):
            Prediction(
                label="simple", scores={"simple": 1.0}, confidence=confidence, classifier="x"
            )

    def test_rejects_label_missing_from_scores(self):
        with pytest.raises(ValueError, match="not among scores"):
            Prediction(label="ghost", scores={"simple": 1.0}, confidence=0.5, classifier="x")

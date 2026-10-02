from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from dumbwaiter.errors import ConfigError
from dumbwaiter.pricing import PriceTable
from dumbwaiter.types import ModelRef, TokenPrices, Usage

USAGE = Usage(input_tokens=1_000_000, output_tokens=100_000)


def row(input_per_mtok: str = "1", output_per_mtok: str = "2") -> TokenPrices:
    return TokenPrices(
        input_per_mtok=Decimal(input_per_mtok),
        output_per_mtok=Decimal(output_per_mtok),
        source="https://example.test/pricing",
        checked_on=date(2026, 1, 1),
    )


class TestNoneAndZeroAreDifferentFacts:
    def test_local_model_costs_a_real_zero(self):
        cost = PriceTable().cost_usd(ModelRef.parse("ollama/llama3.1:latest"), USAGE)
        assert cost == Decimal(0)
        assert cost is not None

    def test_unpriced_cloud_model_costs_none(self):
        assert PriceTable().cost_usd(ModelRef.parse("anthropic/claude-unknown"), USAGE) is None

    def test_bedrock_ships_unpriced(self):
        # No verified Bedrock ids or partner prices yet; a guess would be worse than None.
        assert PriceTable().cost_usd(ModelRef.parse("bedrock/anything"), USAGE) is None


class TestDefaults:
    def test_priced_model_uses_its_row(self):
        # Sonnet 5: $2 in / $10 out per MTok -> 1M in + 0.1M out = $2 + $1.
        cost = PriceTable().cost_usd(ModelRef.parse("anthropic/claude-sonnet-5"), USAGE)
        assert cost == Decimal(3)

    def test_every_default_row_has_provenance(self):
        table = PriceTable()
        for ref, prices in table._prices.items():
            assert prices.source.startswith("https://"), ref
            assert prices.checked_on <= date.today(), ref

    def test_pinned_id_and_alias_price_the_same(self):
        table = PriceTable()
        alias = table.price_for(ModelRef.parse("anthropic/claude-haiku-4-5"))
        pinned = table.price_for(ModelRef.parse("anthropic/claude-haiku-4-5-20251001"))
        assert alias == pinned


class TestOverrides:
    def test_with_prices_adds_without_mutating_the_original(self):
        base = PriceTable()
        extended = base.with_prices({"bedrock/some-model": row("3", "15")})
        ref = ModelRef.parse("bedrock/some-model")
        assert extended.cost_usd(ref, USAGE) == Decimal("4.5")
        assert base.cost_usd(ref, USAGE) is None

    def test_with_prices_replaces_an_existing_row(self):
        ref = ModelRef.parse("anthropic/claude-sonnet-5")
        table = PriceTable().with_prices({str(ref): row("1", "1")})
        assert table.cost_usd(ref, USAGE) == Decimal("1.1")

    def test_explicit_empty_table_prices_nothing(self):
        assert PriceTable({}).cost_usd(ModelRef.parse("anthropic/claude-sonnet-5"), USAGE) is None

    def test_local_providers_are_configurable(self):
        table = PriceTable({}, local_providers={"llamacpp"})
        assert table.cost_usd(ModelRef.parse("llamacpp/x"), USAGE) == Decimal(0)
        assert table.cost_usd(ModelRef.parse("ollama/x"), USAGE) is None

    def test_malformed_spec_is_rejected(self):
        with pytest.raises(ConfigError):
            PriceTable({"no-provider": row()})

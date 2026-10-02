"""Per-model prices, with provenance.

Three answers, kept distinct:

- a local provider (Ollama) costs a real ``0``;
- a model with a price row costs what the row says;
- anything else costs ``None``, because we do not know. Never a plausible-looking guess.

Every row records where it came from and when, because prices drift and partner pricing
(Bedrock) diverges from first-party. Bedrock ships with no rows: its ids come from a real
account and its prices from AWS's own page, neither of which this repo has checked yet.
Callers who have can add them with :meth:`PriceTable.with_prices`.

Costs cover uncached input and output tokens only. A request that opts into prompt
caching through ``Request.extra`` has its cache reads and writes left out of the figure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal

from dumbwaiter.types import ModelRef, TokenPrices, Usage

LOCAL_PROVIDERS = frozenset({"ollama"})
"""Providers that run on the caller's own hardware and bill nothing per token."""

_ANTHROPIC_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
_ANTHROPIC_CHECKED = date(2026, 10, 2)


def _anthropic(input_per_mtok: str, output_per_mtok: str) -> TokenPrices:
    return TokenPrices(
        input_per_mtok=Decimal(input_per_mtok),
        output_per_mtok=Decimal(output_per_mtok),
        source=_ANTHROPIC_SOURCE,
        checked_on=_ANTHROPIC_CHECKED,
    )


# Ids from https://platform.claude.com/docs/en/about-claude/models/overview (current
# models, pinned id and alias) and each legacy model's own overview page.
_DEFAULT_PRICES: dict[str, TokenPrices] = {
    "anthropic/claude-fable-5-1": _anthropic("10", "50"),
    "anthropic/claude-opus-5-5": _anthropic("4", "20"),
    "anthropic/claude-opus-5": _anthropic("5", "25"),
    "anthropic/claude-sonnet-5-5": _anthropic("2", "10"),
    "anthropic/claude-sonnet-5": _anthropic("2", "10"),
    "anthropic/claude-haiku-4-5": _anthropic("1", "5"),
    "anthropic/claude-haiku-4-5-20251001": _anthropic("1", "5"),
}


class PriceTable:
    def __init__(
        self,
        prices: Mapping[str, TokenPrices] | None = None,
        *,
        local_providers: Iterable[str] = LOCAL_PROVIDERS,
    ) -> None:
        """``prices`` is keyed by model spec, e.g. ``anthropic/claude-sonnet-5``."""
        source = _DEFAULT_PRICES if prices is None else prices
        self._prices = {ModelRef.parse(spec): row for spec, row in source.items()}
        self._local = frozenset(local_providers)

    def with_prices(self, prices: Mapping[str, TokenPrices]) -> PriceTable:
        """A copy with ``prices`` added, replacing any rows for the same models."""
        merged = {str(ref): row for ref, row in self._prices.items()}
        merged.update(prices)
        return PriceTable(merged, local_providers=self._local)

    def price_for(self, model: ModelRef) -> TokenPrices | None:
        return self._prices.get(model)

    def cost_usd(self, model: ModelRef, usage: Usage) -> Decimal | None:
        if model.provider in self._local:
            return Decimal(0)
        prices = self.price_for(model)
        return None if prices is None else prices.cost(usage)

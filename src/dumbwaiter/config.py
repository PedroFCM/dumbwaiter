"""Configuration and its validation.

Config is parsed from untrusted YAML, so this is the one place pydantic earns its keep.
Everything that can be caught at load time is caught at load time — a router that only
discovers a typo'd route name on the request that needed it is worse than useless.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from dumbwaiter.errors import ConfigError
from dumbwaiter.types import ModelRef

# Below this, an embedding centroid is being computed from too few points to mean anything.
MIN_EXAMPLES_FOR_EMBEDDING = 3


class RouteConfig(BaseModel):
    """One named destination: a label, the model behind it, and how to recognize it."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    name: str
    model: str
    """Provider-qualified spec, e.g. ``bedrock/amazon.nova-lite-v1:0``."""
    tier: int | None = None
    """Optional ordinal rank, cheap to expensive. Unlocks cost caps and escalation."""
    examples: tuple[str, ...] = ()
    """Prompts that belong on this route. Used to fit the embedding classifier."""
    description: str | None = None
    """Natural-language route description, used by the LLM-judge classifier."""
    max_tokens: int | None = None
    """Per-route override of the request's max_tokens."""

    @field_validator("name")
    @classmethod
    def _name_is_usable(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("route name must not be blank")
        if any(char.isspace() for char in stripped):
            raise ValueError(f"route name {value!r} must not contain whitespace")
        return stripped

    @field_validator("tier")
    @classmethod
    def _tier_is_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError(f"tier must be >= 0, got {value}")
        return value

    @field_validator("model")
    @classmethod
    def _model_spec_parses(cls, value: str) -> str:
        ModelRef.parse(value)  # raises ConfigError on a malformed spec
        return value

    @property
    def model_ref(self) -> ModelRef:
        return ModelRef.parse(self.model)


class ClassifierConfig(BaseModel):
    """Which classification strategy to use, plus its own knobs.

    ``extra="allow"`` because each strategy has a different set of options; they are
    validated by the strategy that consumes them, not here.
    """

    model_config = ConfigDict(frozen=True, extra="allow", protected_namespaces=())

    type: str = "embedding"

    @property
    def options(self) -> dict[str, Any]:
        """Everything except ``type``, handed to the classifier at construction."""
        return dict(self.model_extra or {})


class PolicyConfig(BaseModel):
    """What the router does when things go wrong or get expensive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    escalate_on_error: bool = True
    """On a provider failure, retry one tier up instead of raising."""
    escalate_on_refusal: bool = True
    """Treat ``stop_reason == "refusal"`` as a reason to escalate."""
    max_escalations: int = Field(default=2, ge=0)
    max_cost_per_request_usd: Decimal | None = Field(default=None, gt=0)
    """Refuse to select a route whose estimated cost exceeds this."""


class RouterConfig(BaseModel):
    """The whole configuration, validated as a unit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    routes: tuple[RouteConfig, ...]
    default_route: str
    """Used when confidence is below ``min_confidence`` or the classifier abstains."""
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    log_path: Path | None = None
    """Where decision records are appended as JSONL. None disables logging."""

    @field_validator("routes")
    @classmethod
    def _routes_present_and_unique(cls, value: tuple[RouteConfig, ...]) -> tuple[RouteConfig, ...]:
        if not value:
            raise ValueError("at least one route is required")
        counts = Counter(r.name for r in value)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate route names: {duplicates}")
        return value

    @model_validator(mode="after")
    def _default_route_exists(self) -> Self:
        if self.default_route not in self.route_names:
            raise ValueError(
                f"default_route {self.default_route!r} is not a declared route; "
                f"have {sorted(self.route_names)}"
            )
        return self

    @model_validator(mode="after")
    def _tiers_are_all_or_nothing_and_contiguous(self) -> Self:
        tiered = [r for r in self.routes if r.tier is not None]
        if not tiered:
            return self
        if len(tiered) != len(self.routes):
            untiered = sorted(r.name for r in self.routes if r.tier is None)
            raise ValueError(
                "tier must be set on every route or on none; missing on "
                f"{untiered}. Tiers are what make escalation and cost caps meaningful."
            )
        distinct = sorted({r.tier for r in tiered if r.tier is not None})
        expected = list(range(len(distinct)))
        if distinct != expected:
            raise ValueError(
                f"tiers must form a contiguous range starting at 0; got {distinct}. "
                "Duplicate tiers are fine (several routes can share a rung)."
            )
        return self

    @model_validator(mode="after")
    def _embedding_classifier_has_enough_examples(self) -> Self:
        if self.classifier.type != "embedding":
            return self
        thin = {
            r.name: len(r.examples)
            for r in self.routes
            if len(r.examples) < MIN_EXAMPLES_FOR_EMBEDDING
        }
        if thin:
            raise ValueError(
                f"the embedding classifier needs >= {MIN_EXAMPLES_FOR_EMBEDDING} examples "
                f"per route to build a usable centroid; too few on {thin}"
            )
        return self

    @property
    def route_names(self) -> frozenset[str]:
        return frozenset(r.name for r in self.routes)

    def route(self, name: str) -> RouteConfig:
        for candidate in self.routes:
            if candidate.name == name:
                return candidate
        raise ConfigError(f"unknown route {name!r}; have {sorted(self.route_names)}")

    @property
    def is_tiered(self) -> bool:
        return all(r.tier is not None for r in self.routes)

    @property
    def ladder(self) -> tuple[RouteConfig, ...]:
        """Routes cheapest-first. Falls back to declaration order when untiered."""
        if not self.is_tiered:
            return self.routes
        return tuple(sorted(self.routes, key=lambda r: (r.tier or 0, r.name)))

    def routes_above(self, name: str) -> tuple[RouteConfig, ...]:
        """Escalation candidates: routes on a strictly higher tier, cheapest first."""
        if not self.is_tiered:
            return ()
        current = self.route(name).tier or 0
        return tuple(r for r in self.ladder if (r.tier or 0) > current)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouterConfig:
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> RouterConfig:
        resolved = Path(path)
        try:
            raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {resolved}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"{resolved} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{resolved} must contain a YAML mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)

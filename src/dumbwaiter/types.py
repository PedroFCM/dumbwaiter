"""Core value types.

These are plain frozen dataclasses rather than pydantic models on purpose: they sit on
the hot path (every request builds a Request and a Decision) and they are constructed by
us, not parsed from untrusted input. Config is the opposite case and does use pydantic.

Design note — the normalized request surface is deliberately small. Current Anthropic
models reject ``temperature``, ``top_p``, ``top_k`` and ``budget_tokens`` outright, so a
generic "temperature" field that got forwarded to every provider would be a 400 waiting
to happen. Shared fields only here; anything provider-specific goes in ``Request.extra``
and is merged by the one provider that understands it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from dumbwaiter.errors import ConfigError

Role = Literal["user", "assistant"]

# Separates provider from model id in a model spec, e.g. "bedrock/amazon.nova-lite-v1:0".
_MODEL_SEP = "/"


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A provider-qualified model identifier.

    Model ids routinely contain dots and colons (``amazon.nova-lite-v1:0``), so the spec
    is split on the *first* separator only.
    """

    provider: str
    model_id: str

    @classmethod
    def parse(cls, spec: str) -> ModelRef:
        provider, sep, model_id = spec.partition(_MODEL_SEP)
        if not sep or not provider or not model_id:
            raise ConfigError(
                f"model spec {spec!r} must be '<provider>/<model-id>', "
                f"e.g. 'bedrock/amazon.nova-lite-v1:0' or 'anthropic/claude-sonnet-5'"
            )
        return cls(provider=provider, model_id=model_id)

    def __str__(self) -> str:
        return f"{self.provider}{_MODEL_SEP}{self.model_id}"


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of conversation. Text-only in v1; multimodal is a later concern."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Request:
    """A normalized completion request, before a model has been chosen."""

    messages: Sequence[Message]
    system: str | None = None
    max_tokens: int = 4096
    stop: Sequence[str] = ()
    force_route: str | None = None
    """Bypass the classifier and use this route by name. Recorded on the Decision."""
    extra: Mapping[str, Any] = field(default_factory=dict)
    """Provider-specific parameters, keyed by provider name.

    ``{"ollama": {"options": {"temperature": 0.2}}, "anthropic": {...}}``. A request does
    not know in advance which route it will land on, so each provider reads only its own
    key and ignores the rest.
    """

    def extra_for(self, provider: str) -> Mapping[str, Any]:
        """The parameters addressed to one provider, or an empty mapping."""
        value = self.extra.get(provider, {})
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Request.extra[{provider!r}] must be a mapping, got {type(value).__name__}"
            )
        return value

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("Request.messages must not be empty")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    @classmethod
    def user(cls, text: str, **kwargs: Any) -> Request:
        """Shorthand for the single-turn case."""
        return cls(messages=[Message(role="user", content=text)], **kwargs)

    @property
    def routing_text(self) -> str:
        """The text the classifier sees.

        The last user turn, because that is what the request is actually asking for.
        Earlier turns describe what was already answered and would blur the signal.
        """
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one completion."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class Completion:
    """What a provider returns: the model's answer, before routing context is attached.

    The router turns this into a :class:`Response` by adding the decision, latency, and
    cost. Providers stay ignorant of all three.
    """

    text: str
    usage: Usage
    stop_reason: str | None = None
    """Normalized where a provider has an obvious equivalent: ``end_turn``,
    ``max_tokens``, ``stop_sequence``, ``tool_use``, ``refusal``. Anything else is passed
    through verbatim rather than squeezed into a category it may not belong to."""
    raw: Any = None


@dataclass(frozen=True, slots=True)
class TokenPrices:
    """Price per million tokens, with provenance.

    ``source`` and ``checked_on`` exist because partner pricing (Bedrock) diverges from
    first-party pricing and both drift. A price with no provenance is a bug report
    waiting to happen.
    """

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    source: str
    checked_on: date

    def cost(self, usage: Usage) -> Decimal:
        per_mtok = Decimal(1_000_000)
        return (
            self.input_per_mtok * Decimal(usage.input_tokens)
            + self.output_per_mtok * Decimal(usage.output_tokens)
        ) / per_mtok


@dataclass(frozen=True, slots=True)
class Prediction:
    """What a classifier concluded about a request."""

    label: str
    scores: Mapping[str, float]
    confidence: float
    classifier: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.scores and self.label not in self.scores:
            raise ValueError(f"label {self.label!r} is not among scores {sorted(self.scores)}")

    @property
    def runner_up(self) -> str | None:
        """The second-best label, or None when there is nothing to compare against."""
        ranked = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[1][0] if len(ranked) > 1 else None


@dataclass(frozen=True, slots=True)
class Decision:
    """Why a request is going where it is going.

    Returned by ``Router.route()`` without executing anything. This is the transparency
    feature: a caller can inspect, log, or override the decision before paying for it.
    """

    route: str
    model: ModelRef
    prediction: Prediction
    reason: str
    fallback_chain: Sequence[ModelRef] = ()
    overridden: bool = False


@dataclass(frozen=True, slots=True)
class Response:
    """A completed request, with what it cost and how it got here."""

    text: str
    model: ModelRef
    usage: Usage
    decision: Decision
    latency_ms: float
    stop_reason: str | None = None
    cost_usd: Decimal | None = None
    """None when the model has no known price entry. Never a guessed number."""
    raw: Any = None
    """The untouched provider response, for callers that need something we dropped."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """One streamed fragment. ``usage`` and ``stop_reason`` arrive on the final chunk."""

    text: str = ""
    usage: Usage | None = None
    stop_reason: str | None = None

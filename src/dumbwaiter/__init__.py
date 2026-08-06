"""dumbwaiter — cost-aware LLM routing.

Most requests do not need the smart model. dumbwaiter classifies each request and
sends it to the cheapest route that can answer it, across AWS Bedrock and Anthropic.
"""

from __future__ import annotations

from dumbwaiter.config import ClassifierConfig, PolicyConfig, RouteConfig, RouterConfig
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
from dumbwaiter.types import (
    Chunk,
    Decision,
    Message,
    ModelRef,
    Prediction,
    Request,
    Response,
    TokenPrices,
    Usage,
)

__version__ = "0.0.1"

__all__ = [
    "BudgetExceeded",
    "Chunk",
    "ClassifierConfig",
    "ClassifierError",
    "ConfigError",
    "Decision",
    "DumbwaiterError",
    "Message",
    "ModelRef",
    "NoRouteError",
    "PolicyConfig",
    "Prediction",
    "ProviderError",
    "ProviderTimeout",
    "RateLimited",
    "Request",
    "Response",
    "RouteConfig",
    "RouterConfig",
    "RoutingError",
    "TokenPrices",
    "Usage",
    "__version__",
]

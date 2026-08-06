from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from dumbwaiter.config import RouterConfig
from dumbwaiter.errors import ConfigError

EXAMPLE_CONFIG = Path(__file__).parent.parent / "examples" / "config.yaml"


def _examples(count: int = 3) -> list[str]:
    return [f"example prompt number {i}" for i in range(count)]


def base_config(**overrides: Any) -> dict[str, Any]:
    """A minimal valid config, so each test can break exactly one thing."""
    data: dict[str, Any] = {
        "default_route": "simple",
        "classifier": {"type": "embedding", "backend": "bedrock"},
        "routes": [
            {
                "name": "simple",
                "model": "bedrock/amazon.nova-lite-v1:0",
                "tier": 0,
                "examples": _examples(),
            },
            {
                "name": "complex",
                "model": "anthropic/claude-opus-5",
                "tier": 1,
                "examples": _examples(),
            },
        ],
    }
    data.update(overrides)
    return data


class TestValidConfig:
    def test_loads_and_exposes_routes(self):
        config = RouterConfig.from_dict(base_config())
        assert config.route_names == {"simple", "complex"}
        assert config.route("simple").model_ref.provider == "bedrock"

    def test_shipped_example_config_is_valid(self):
        # Guards against the README example rotting out of sync with validation.
        config = RouterConfig.from_yaml(EXAMPLE_CONFIG)
        assert config.is_tiered
        assert config.default_route in config.route_names

    def test_classifier_options_exclude_the_type_field(self):
        config = RouterConfig.from_dict(base_config())
        assert config.classifier.type == "embedding"
        assert config.classifier.options == {"backend": "bedrock"}

    def test_policy_defaults_are_conservative(self):
        config = RouterConfig.from_dict(base_config())
        assert config.policy.escalate_on_error is True
        assert config.policy.max_cost_per_request_usd is None

    def test_cost_cap_parses_as_decimal(self):
        config = RouterConfig.from_dict(base_config(policy={"max_cost_per_request_usd": "0.05"}))
        assert config.policy.max_cost_per_request_usd == Decimal("0.05")


class TestLadderAndEscalation:
    def test_ladder_is_ordered_cheapest_first(self):
        data = base_config()
        data["routes"] = list(reversed(data["routes"]))  # declaration order is wrong on purpose
        config = RouterConfig.from_dict(data)
        assert [r.name for r in config.ladder] == ["simple", "complex"]

    def test_routes_above_returns_higher_tiers_only(self):
        config = RouterConfig.from_dict(base_config())
        assert [r.name for r in config.routes_above("simple")] == ["complex"]
        assert config.routes_above("complex") == ()

    def test_untiered_config_has_no_escalation_path(self):
        data = base_config()
        for route in data["routes"]:
            route.pop("tier")
        config = RouterConfig.from_dict(data)
        assert config.is_tiered is False
        assert config.routes_above("simple") == ()

    def test_untiered_ladder_preserves_declaration_order(self):
        # Without tiers there is no cost ordering to infer, so declaration order is the
        # only honest answer — do not silently sort by name.
        data = base_config()
        for route in data["routes"]:
            route.pop("tier")
        data["routes"] = list(reversed(data["routes"]))
        config = RouterConfig.from_dict(data)
        assert [r.name for r in config.ladder] == ["complex", "simple"]

    def test_shared_tiers_are_allowed(self):
        # Several routes can sit on the same rung: `code` and `vision` are both mid-tier.
        data = base_config()
        data["routes"].append(
            {
                "name": "vision",
                "model": "anthropic/claude-sonnet-5",
                "tier": 1,
                "examples": _examples(),
            }
        )
        config = RouterConfig.from_dict(data)
        assert len(config.routes_above("simple")) == 2


class TestRouteNameValidation:
    """Route names end up as dict keys, log fields, and `force_route` values, so they
    have to be usable as identifiers rather than free text."""

    def test_strips_surrounding_whitespace(self):
        data = base_config()
        data["routes"][0]["name"] = "  simple  "
        config = RouterConfig.from_dict(data)
        assert config.route("simple").name == "simple"

    @pytest.mark.parametrize("name", ["", "   ", "\t"])
    def test_rejects_blank_names(self, name: str):
        data = base_config()
        data["routes"][0]["name"] = name
        with pytest.raises(ConfigError, match="must not be blank"):
            RouterConfig.from_dict(data)

    @pytest.mark.parametrize("name", ["two words", "tab\tname", "line\nbreak"])
    def test_rejects_internal_whitespace(self, name: str):
        data = base_config()
        data["routes"][0]["name"] = name
        with pytest.raises(ConfigError, match="must not contain whitespace"):
            RouterConfig.from_dict(data)

    def test_rejects_negative_tier(self):
        data = base_config()
        data["routes"][0]["tier"] = -1
        with pytest.raises(ConfigError, match="tier must be >= 0"):
            RouterConfig.from_dict(data)


class TestInvalidConfig:
    def test_rejects_empty_routes(self):
        with pytest.raises(ConfigError, match="at least one route"):
            RouterConfig.from_dict(base_config(routes=[]))

    def test_rejects_duplicate_route_names(self):
        data = base_config()
        data["routes"][1]["name"] = "simple"
        with pytest.raises(ConfigError, match="duplicate route names"):
            RouterConfig.from_dict(data)

    def test_rejects_unknown_default_route(self):
        with pytest.raises(ConfigError, match="not a declared route"):
            RouterConfig.from_dict(base_config(default_route="nope"))

    def test_rejects_malformed_model_spec(self):
        data = base_config()
        data["routes"][0]["model"] = "amazon.nova-lite-v1:0"  # missing provider prefix
        with pytest.raises(ConfigError, match="<provider>/<model-id>"):
            RouterConfig.from_dict(data)

    def test_rejects_partially_tiered_routes(self):
        data = base_config()
        data["routes"][1].pop("tier")
        with pytest.raises(ConfigError, match="every route or on none"):
            RouterConfig.from_dict(data)

    def test_rejects_tier_gaps(self):
        data = base_config()
        data["routes"][1]["tier"] = 5
        with pytest.raises(ConfigError, match="contiguous range"):
            RouterConfig.from_dict(data)

    def test_rejects_too_few_examples_for_embedding_classifier(self):
        data = base_config()
        data["routes"][0]["examples"] = ["only one"]
        with pytest.raises(ConfigError, match="usable centroid"):
            RouterConfig.from_dict(data)

    def test_allows_thin_examples_for_a_non_embedding_classifier(self):
        # A rules classifier does not need examples at all.
        data = base_config(classifier={"type": "rules"})
        data["routes"][0]["examples"] = []
        config = RouterConfig.from_dict(data)
        assert config.classifier.type == "rules"

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_rejects_out_of_range_min_confidence(self, value: float):
        with pytest.raises(ConfigError):
            RouterConfig.from_dict(base_config(min_confidence=value))

    def test_rejects_unknown_top_level_keys(self):
        # extra="forbid" turns a typo into a load-time error instead of silence.
        with pytest.raises(ConfigError):
            RouterConfig.from_dict(base_config(mim_confidence=0.2))

    def test_rejects_non_positive_cost_cap(self):
        with pytest.raises(ConfigError):
            RouterConfig.from_dict(base_config(policy={"max_cost_per_request_usd": 0}))

    def test_route_lookup_of_unknown_name_raises(self):
        config = RouterConfig.from_dict(base_config())
        with pytest.raises(ConfigError, match="unknown route"):
            config.route("ghost")


class TestYamlLoading:
    def test_missing_file_reports_the_path(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            RouterConfig.from_yaml(tmp_path / "absent.yaml")

    def test_malformed_yaml_is_reported_as_config_error(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("routes: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            RouterConfig.from_yaml(bad)

    def test_non_mapping_yaml_is_rejected(self, tmp_path: Path):
        scalar = tmp_path / "scalar.yaml"
        scalar.write_text("just a string", encoding="utf-8")
        with pytest.raises(ConfigError, match="must contain a YAML mapping"):
            RouterConfig.from_yaml(scalar)

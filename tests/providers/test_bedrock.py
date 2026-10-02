"""Bedrock provider, against botocore's Stubber.

Stubber validates both the parameters we send and the responses we fake against the
real ``bedrock-runtime`` service model, so a malformed Converse call fails here.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import boto3
import pytest
from botocore.exceptions import (
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
)
from botocore.stub import Stubber

from dumbwaiter.errors import ConfigError, ProviderError, ProviderTimeout, RateLimited
from dumbwaiter.providers import BedrockProvider, Provider, ProviderRegistry
from dumbwaiter.types import Message, Request

# Any string satisfies the stub; this is not a claim about what exists in an account.
MODEL = "amazon.nova-lite-v1:0"


def converse_response(**overrides: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "output": {"message": {"role": "assistant", "content": [{"text": "Hello."}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 12, "outputTokens": 3, "totalTokens": 15},
        "metrics": {"latencyMs": 210},
    }
    response.update(overrides)
    return response


def expected(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "modelId": MODEL,
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "inferenceConfig": {"maxTokens": 4096},
    }
    params.update(overrides)
    return params


@pytest.fixture
def client() -> Any:
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


@pytest.fixture
def stubber(client: Any) -> Iterator[Stubber]:
    with Stubber(client) as stub:
        yield stub
        stub.assert_no_pending_responses()


@pytest.fixture
def provider(client: Any) -> BedrockProvider:
    return BedrockProvider(client)


class RaisingClient:
    """Stubber only fakes ClientError; transport failures need a client that raises."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def converse(self, **_: Any) -> dict[str, Any]:
        raise self.error

    def close(self) -> None:
        pass


class TestConstruction:
    def test_satisfies_the_provider_protocol(self, provider):
        assert isinstance(provider, Provider)

    def test_registered_by_default(self):
        assert "bedrock" in ProviderRegistry().names

    def test_builds_its_own_client(self):
        provider = BedrockProvider(region_name="us-east-1")
        assert provider._client.meta.region_name == "us-east-1"

    def test_missing_region_is_a_config_error(self, monkeypatch):
        for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
        with pytest.raises(ConfigError, match="cannot build a Bedrock client"):
            BedrockProvider()

    def test_missing_sdk_names_the_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "boto3", None)
        with pytest.raises(ImportError, match=r"pip install dumbwaiter\[bedrock\]"):
            BedrockProvider()


class TestCompletion:
    async def test_maps_text_usage_and_stop_reason(self, provider, stubber):
        stubber.add_response("converse", converse_response(), expected())
        completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.text == "Hello."
        assert completion.usage.input_tokens == 12
        assert completion.usage.output_tokens == 3
        assert completion.stop_reason == "end_turn"
        assert completion.raw["metrics"]["latencyMs"] == 210

    async def test_only_text_blocks_make_the_answer(self, provider, stubber):
        content = [
            {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
            {"text": "Part one. "},
            {"text": "Part two."},
        ]
        response = converse_response(output={"message": {"role": "assistant", "content": content}})
        stubber.add_response("converse", response, expected())
        completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.text == "Part one. Part two."

    async def test_bedrock_only_stop_reasons_pass_through(self, provider, stubber):
        response = converse_response(
            output={"message": {"role": "assistant", "content": []}},
            stopReason="content_filtered",
        )
        stubber.add_response("converse", response, expected())
        completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.stop_reason == "content_filtered"
        assert completion.text == ""


class TestPayload:
    async def test_shared_fields_map_onto_converse(self, provider, stubber):
        request = Request(
            messages=[
                Message(role="user", content="a"),
                Message(role="assistant", content="b"),
                Message(role="user", content="c"),
            ],
            system="Be terse.",
            max_tokens=100,
            stop=["END"],
        )
        stubber.add_response(
            "converse",
            converse_response(),
            expected(
                messages=[
                    {"role": "user", "content": [{"text": "a"}]},
                    {"role": "assistant", "content": [{"text": "b"}]},
                    {"role": "user", "content": [{"text": "c"}]},
                ],
                system=[{"text": "Be terse."}],
                inferenceConfig={"maxTokens": 100, "stopSequences": ["END"]},
            ),
        )
        await provider.complete(request, MODEL)

    async def test_own_extra_is_merged(self, provider, stubber):
        request = Request.user(
            "hi",
            extra={
                "bedrock": {
                    "inferenceConfig": {"temperature": 0.2},
                    "additionalModelRequestFields": {"top_k": 50},
                }
            },
        )
        stubber.add_response(
            "converse",
            converse_response(),
            expected(
                inferenceConfig={"maxTokens": 4096, "temperature": 0.2},
                additionalModelRequestFields={"top_k": 50},
            ),
        )
        await provider.complete(request, MODEL)

    async def test_other_providers_extra_is_ignored(self, provider, stubber):
        # Stubber rejects any parameter it was not told to expect.
        stubber.add_response("converse", converse_response(), expected())
        await provider.complete(Request.user("hi", extra={"ollama": {"keep_alive": "5m"}}), MODEL)

    async def test_extra_cannot_override_model_or_messages(self, provider, stubber):
        stubber.add_response("converse", converse_response(), expected())
        request = Request.user("hi", extra={"bedrock": {"modelId": "other", "messages": []}})
        await provider.complete(request, MODEL)

    async def test_non_mapping_inference_config_is_rejected(self, provider):
        request = Request.user("hi", extra={"bedrock": {"inferenceConfig": 1}})
        with pytest.raises(TypeError, match="inferenceConfig"):
            await provider.complete(request, MODEL)


class TestErrors:
    async def test_throttling_is_rate_limited_with_retry_after(self, provider, stubber):
        stubber.add_client_error(
            "converse",
            service_error_code="ThrottlingException",
            service_message="Too many requests",
            http_status_code=429,
            response_meta={"HTTPHeaders": {"retry-after": "4"}},
        )
        with pytest.raises(RateLimited, match="ThrottlingException") as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert caught.value.retry_after == 4.0
        assert caught.value.provider == "bedrock"
        assert caught.value.model == MODEL

    async def test_throttling_without_retry_after(self, provider, stubber):
        stubber.add_client_error("converse", "ThrottlingException", "slow", 429)
        with pytest.raises(RateLimited) as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert caught.value.retry_after is None

    async def test_model_timeout_is_a_provider_timeout(self, provider, stubber):
        stubber.add_client_error("converse", "ModelTimeoutException", "took too long", 408)
        with pytest.raises(ProviderTimeout, match="ModelTimeoutException"):
            await provider.complete(Request.user("hi"), MODEL)

    @pytest.mark.parametrize(
        "code", ["AccessDeniedException", "ValidationException", "ResourceNotFoundException"]
    )
    async def test_other_service_errors_carry_the_code(self, provider, stubber, code):
        stubber.add_client_error("converse", code, "nope", 400)
        with pytest.raises(ProviderError, match=f"{code}: nope") as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert type(caught.value) is ProviderError

    async def test_error_without_a_message_reports_the_code(self, provider, stubber):
        stubber.add_client_error("converse", "InternalServerException", "", 500)
        with pytest.raises(ProviderError, match=r"\] InternalServerException$"):
            await provider.complete(Request.user("hi"), MODEL)

    async def test_read_timeout_is_a_provider_timeout(self):
        provider = BedrockProvider(RaisingClient(ReadTimeoutError(endpoint_url="https://x")))
        with pytest.raises(ProviderTimeout):
            await provider.complete(Request.user("hi"), MODEL)

    @pytest.mark.parametrize(
        "error", [NoCredentialsError(), EndpointConnectionError(endpoint_url="https://x")]
    )
    async def test_client_side_failures_are_provider_errors(self, error):
        provider = BedrockProvider(RaisingClient(error))
        with pytest.raises(ProviderError) as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert type(caught.value) is ProviderError


class TestLifecycle:
    async def test_aclose_is_idempotent(self, provider):
        await provider.aclose()
        await provider.aclose()

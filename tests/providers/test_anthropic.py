"""Anthropic provider, against a real AsyncAnthropic client on an httpx2 MockTransport.

``anthropic`` 1.x runs on httpx2, which respx does not patch, so the fake sits at the
transport. That keeps the SDK's own response parsing and error mapping in the test.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from dumbwaiter.errors import ProviderError, ProviderTimeout, RateLimited
from dumbwaiter.providers import AnthropicProvider, Provider, ProviderRegistry
from dumbwaiter.types import Message, Request

MODEL = "claude-sonnet-5"


def message_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }
    body.update(overrides)
    return body


def error_body(kind: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


class Recorder:
    """A transport handler that answers with one canned response and keeps the request."""

    def __init__(self, response: httpx2.Response | Exception) -> None:
        self.response = response
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    @property
    def payload(self) -> dict[str, Any]:
        sent: dict[str, Any] = json.loads(self.requests[-1].content)
        return sent


@pytest.fixture
def make_provider() -> Callable[[Recorder], AnthropicProvider]:
    def build(handler: Recorder) -> AnthropicProvider:
        client = anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,  # the SDK would otherwise retry 429/5xx against the fake
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        return AnthropicProvider(client)

    return build


class TestProtocol:
    def test_satisfies_the_provider_protocol(self):
        assert isinstance(AnthropicProvider(api_key="test-key"), Provider)

    def test_registered_by_default(self):
        assert "anthropic" in ProviderRegistry().names

    def test_missing_sdk_names_the_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "anthropic", None)
        with pytest.raises(ImportError, match=r"pip install dumbwaiter\[anthropic\]"):
            AnthropicProvider()


class TestCompletion:
    async def test_maps_text_usage_and_stop_reason(self, make_provider):
        provider = make_provider(Recorder(httpx2.Response(200, json=message_body())))
        completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.text == "Hello."
        assert completion.usage.input_tokens == 12
        assert completion.usage.output_tokens == 3
        assert completion.stop_reason == "end_turn"
        assert completion.raw.id == "msg_test"

    async def test_only_text_blocks_make_the_answer(self, make_provider):
        body = message_body(
            content=[
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Part one. "},
                {"type": "text", "text": "Part two."},
            ]
        )
        provider = make_provider(Recorder(httpx2.Response(200, json=body)))
        completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.text == "Part one. Part two."

    async def test_refusal_stop_reason_survives(self, make_provider):
        # Escalation policy keys on exactly this value.
        body = message_body(content=[], stop_reason="refusal")
        provider = make_provider(Recorder(httpx2.Response(200, json=body)))
        completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.stop_reason == "refusal"
        assert completion.text == ""


class TestPayload:
    async def test_shared_fields_map_onto_the_messages_api(self, make_provider):
        recorder = Recorder(httpx2.Response(200, json=message_body()))
        provider = make_provider(recorder)
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
        await provider.complete(request, MODEL)
        assert recorder.payload == {
            "model": MODEL,
            "max_tokens": 100,
            "system": "Be terse.",
            "stop_sequences": ["END"],
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
        }

    async def test_unset_optional_fields_are_omitted(self, make_provider):
        recorder = Recorder(httpx2.Response(200, json=message_body()))
        await make_provider(recorder).complete(Request.user("hi"), MODEL)
        assert "system" not in recorder.payload
        assert "stop_sequences" not in recorder.payload

    async def test_own_extra_is_merged_into_the_body(self, make_provider):
        recorder = Recorder(httpx2.Response(200, json=message_body()))
        request = Request.user("hi", extra={"anthropic": {"output_config": {"effort": "low"}}})
        await make_provider(recorder).complete(request, MODEL)
        assert recorder.payload["output_config"] == {"effort": "low"}

    async def test_other_providers_extra_is_ignored(self, make_provider):
        recorder = Recorder(httpx2.Response(200, json=message_body()))
        request = Request.user("hi", extra={"ollama": {"options": {"temperature": 0.2}}})
        await make_provider(recorder).complete(request, MODEL)
        assert "options" not in recorder.payload
        assert "temperature" not in recorder.payload


class TestErrors:
    async def test_429_is_rate_limited_with_retry_after(self, make_provider):
        response = httpx2.Response(
            429, json=error_body("rate_limit_error", "slow down"), headers={"retry-after": "12"}
        )
        provider = make_provider(Recorder(response))
        with pytest.raises(RateLimited) as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert caught.value.retry_after == 12.0
        assert caught.value.provider == "anthropic"
        assert caught.value.model == MODEL

    async def test_429_without_retry_after(self, make_provider):
        response = httpx2.Response(429, json=error_body("rate_limit_error", "slow down"))
        provider = make_provider(Recorder(response))
        with pytest.raises(RateLimited) as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert caught.value.retry_after is None

    @pytest.mark.parametrize(
        ("status", "kind"),
        [(400, "invalid_request_error"), (404, "not_found_error"), (529, "overloaded_error")],
    )
    async def test_status_errors_carry_the_code(self, make_provider, status, kind):
        response = httpx2.Response(status, json=error_body(kind, "nope"))
        provider = make_provider(Recorder(response))
        with pytest.raises(ProviderError, match=f"HTTP {status}") as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert type(caught.value) is ProviderError

    async def test_timeout_is_a_provider_timeout(self, make_provider):
        provider = make_provider(Recorder(httpx2.ReadTimeout("slow")))
        with pytest.raises(ProviderTimeout):
            await provider.complete(Request.user("hi"), MODEL)

    async def test_connection_failure_is_a_provider_error(self, make_provider):
        provider = make_provider(Recorder(httpx2.ConnectError("refused")))
        with pytest.raises(ProviderError) as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert type(caught.value) is ProviderError


class TestLifecycle:
    async def test_aclose_is_idempotent(self, make_provider):
        provider = make_provider(Recorder(httpx2.Response(200, json=message_body())))
        await provider.aclose()
        await provider.aclose()

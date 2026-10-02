"""Ollama provider, against respx stubs shaped like real Ollama 0.34.4 responses."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import respx

from dumbwaiter.errors import ProviderError, ProviderTimeout, RateLimited
from dumbwaiter.providers import OllamaProvider, Provider
from dumbwaiter.types import Message, Request

BASE = "http://ollama.test"
MODEL = "llama3.1:latest"


def chat_body(content: str = "Hello how are you?", **overrides: Any) -> dict[str, Any]:
    # Captured from a live `POST /api/chat` with stream=false.
    body: dict[str, Any] = {
        "model": MODEL,
        "created_at": "2026-10-02T14:44:02.404527Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "total_duration": 4259085167,
        "prompt_eval_count": 24,
        "eval_count": 6,
    }
    body.update(overrides)
    return body


@pytest.fixture
async def provider() -> AsyncIterator[OllamaProvider]:
    instance = OllamaProvider(BASE)
    yield instance
    await instance.aclose()


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=False) as router:
        yield router


def sent_payload(api: respx.MockRouter) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(api.calls.last.request.content)
    return payload


class TestProtocol:
    def test_satisfies_the_provider_protocol(self):
        assert isinstance(OllamaProvider(BASE), Provider)

    def test_name_matches_the_model_spec_prefix(self):
        assert OllamaProvider.name == "ollama"


class TestCompletion:
    async def test_maps_text_usage_and_stop_reason(self, provider, api):
        api.post("/api/chat").respond(json=chat_body())
        completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.text == "Hello how are you?"
        assert completion.usage.input_tokens == 24
        assert completion.usage.output_tokens == 6
        assert completion.stop_reason == "end_turn"
        assert completion.raw["model"] == MODEL

    async def test_length_is_normalized_to_max_tokens(self, provider, api):
        api.post("/api/chat").respond(json=chat_body(done_reason="length"))
        completion = await provider.complete(Request.user("count to 50"), MODEL)
        assert completion.stop_reason == "max_tokens"

    async def test_unknown_stop_reason_passes_through(self, provider, api):
        api.post("/api/chat").respond(json=chat_body(done_reason="load"))
        completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.stop_reason == "load"

    async def test_missing_counts_default_to_zero(self, provider, api):
        # A "load" response carries no eval counts at all.
        body = chat_body()
        del body["prompt_eval_count"], body["eval_count"]
        api.post("/api/chat").respond(json=body)
        completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.usage.total_tokens == 0


class TestPayload:
    async def test_system_becomes_the_first_message(self, provider, api):
        api.post("/api/chat").respond(json=chat_body())
        request = Request(
            messages=[
                Message(role="user", content="a"),
                Message(role="assistant", content="b"),
                Message(role="user", content="c"),
            ],
            system="Be terse.",
        )
        await provider.complete(request, MODEL)
        payload = sent_payload(api)
        assert payload["model"] == MODEL
        assert payload["stream"] is False
        assert payload["messages"] == [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]

    async def test_no_system_message_when_unset(self, provider, api):
        api.post("/api/chat").respond(json=chat_body())
        await provider.complete(Request.user("hi"), MODEL)
        assert [m["role"] for m in sent_payload(api)["messages"]] == ["user"]

    async def test_max_tokens_and_stop_go_in_options(self, provider, api):
        api.post("/api/chat").respond(json=chat_body())
        await provider.complete(Request.user("hi", max_tokens=20, stop=["\n\n"]), MODEL)
        assert sent_payload(api)["options"] == {"num_predict": 20, "stop": ["\n\n"]}

    async def test_own_extra_is_merged(self, provider, api):
        api.post("/api/chat").respond(json=chat_body())
        request = Request.user(
            "hi",
            extra={"ollama": {"keep_alive": "10m", "options": {"temperature": 0.2}}},
        )
        await provider.complete(request, MODEL)
        payload = sent_payload(api)
        assert payload["keep_alive"] == "10m"
        assert payload["options"]["temperature"] == 0.2
        assert payload["options"]["num_predict"] == 4096

    async def test_other_providers_extra_is_ignored(self, provider, api):
        # A request does not know which route it lands on; Anthropic-only params
        # must never leak into an Ollama call.
        api.post("/api/chat").respond(json=chat_body())
        await provider.complete(Request.user("hi", extra={"anthropic": {"top_k": 5}}), MODEL)
        payload = sent_payload(api)
        assert "anthropic" not in payload
        assert "top_k" not in payload["options"]

    async def test_extra_cannot_override_model_or_messages(self, provider, api):
        api.post("/api/chat").respond(json=chat_body())
        request = Request.user("hi", extra={"ollama": {"model": "other", "stream": True}})
        await provider.complete(request, MODEL)
        payload = sent_payload(api)
        assert payload["model"] == MODEL
        assert payload["stream"] is False

    async def test_non_mapping_options_is_rejected(self, provider):
        with pytest.raises(TypeError, match="options"):
            await provider.complete(Request.user("hi", extra={"ollama": {"options": 1}}), MODEL)


class TestErrors:
    async def test_missing_model_suggests_pulling_it(self, provider, api):
        api.post("/api/chat").respond(404, json={"error": "model 'nope:latest' not found"})
        with pytest.raises(ProviderError, match="ollama pull nope:latest") as caught:
            await provider.complete(Request.user("hi"), "nope:latest")
        assert caught.value.provider == "ollama"
        assert caught.value.model == "nope:latest"

    async def test_daemon_down_is_a_provider_error(self, provider, api):
        api.post("/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="ollama serve"):
            await provider.complete(Request.user("hi"), MODEL)

    async def test_timeout_is_a_provider_timeout(self, provider, api):
        api.post("/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProviderTimeout):
            await provider.complete(Request.user("hi"), MODEL)

    async def test_other_transport_errors_are_wrapped(self, provider, api):
        api.post("/api/chat").mock(side_effect=httpx.RemoteProtocolError("bad"))
        with pytest.raises(ProviderError) as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert type(caught.value) is ProviderError

    async def test_429_is_rate_limited_with_retry_after(self, provider, api):
        api.post("/api/chat").respond(
            429, json={"error": "server busy"}, headers={"Retry-After": "7"}
        )
        with pytest.raises(RateLimited) as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert caught.value.retry_after == 7.0

    async def test_429_without_retry_after(self, provider, api):
        api.post("/api/chat").respond(429, json={"error": "server busy"})
        with pytest.raises(RateLimited) as caught:
            await provider.complete(Request.user("hi"), MODEL)
        assert caught.value.retry_after is None

    async def test_server_error_carries_status_and_detail(self, provider, api):
        api.post("/api/chat").respond(500, json={"error": "out of memory"})
        with pytest.raises(ProviderError, match="HTTP 500: out of memory"):
            await provider.complete(Request.user("hi"), MODEL)

    async def test_non_json_error_body_still_reports(self, provider, api):
        api.post("/api/chat").respond(502, text="Bad Gateway")
        with pytest.raises(ProviderError, match="HTTP 502: Bad Gateway"):
            await provider.complete(Request.user("hi"), MODEL)

    async def test_empty_error_body_falls_back_to_reason(self, provider, api):
        api.post("/api/chat").respond(503)
        with pytest.raises(ProviderError, match="HTTP 503: Service Unavailable"):
            await provider.complete(Request.user("hi"), MODEL)

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, text="not json"),
            httpx.Response(200, json={"done": True}),
            httpx.Response(200, json={"message": None}),
        ],
    )
    async def test_malformed_success_body_is_a_provider_error(self, provider, api, response):
        api.post("/api/chat").mock(return_value=response)
        with pytest.raises(ProviderError, match="unexpected response shape"):
            await provider.complete(Request.user("hi"), MODEL)


class TestLifecycle:
    async def test_uses_an_injected_client(self, api):
        api.post("/api/chat").respond(json=chat_body("from injected"))
        async with httpx.AsyncClient(base_url=BASE) as client:
            provider = OllamaProvider(client=client)
            completion = await provider.complete(Request.user("hi"), MODEL)
        assert completion.text == "from injected"

    async def test_aclose_is_idempotent(self):
        provider = OllamaProvider(BASE)
        await provider.aclose()
        await provider.aclose()

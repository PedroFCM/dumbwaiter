"""Ollama, the default provider: a local daemon, no credentials, no per-token cost.

Talks to ``POST /api/chat`` with ``stream: false``. Response and error shapes were checked
against Ollama 0.34.4.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from dumbwaiter.errors import ProviderError, ProviderTimeout, RateLimited
from dumbwaiter.types import Completion, Request, Usage

DEFAULT_BASE_URL = "http://localhost:11434"

# Generous because the first request to a model pays its load time, which for a
# multi-gigabyte model on a laptop can run well past a minute.
DEFAULT_TIMEOUT_S = 300.0

# Ollama reports "stop" for both a natural end and a matched stop sequence, so it cannot
# be split further. "load" means the model was loaded and nothing was generated.
_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def complete(self, request: Request, model_id: str) -> Completion:
        payload = self._payload(request, model_id)
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"no response within the timeout: {exc}", provider=self.name, model=model_id
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"cannot reach the Ollama daemon at {self._client.base_url}; is `ollama serve` "
                f"running? ({exc})",
                provider=self.name,
                model=model_id,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc), provider=self.name, model=model_id) from exc

        if response.status_code != httpx.codes.OK:
            raise self._status_error(response, model_id)

        try:
            body = response.json()
            text = body["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderError(
                f"unexpected response shape: {response.text[:200]!r}",
                provider=self.name,
                model=model_id,
            ) from exc

        done_reason = body.get("done_reason")
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=body.get("prompt_eval_count", 0),
                output_tokens=body.get("eval_count", 0),
            ),
            stop_reason=_STOP_REASONS.get(done_reason, done_reason),
            raw=body,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, request: Request, model_id: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)

        options: dict[str, Any] = {"num_predict": request.max_tokens}
        if request.stop:
            options["stop"] = list(request.stop)

        # Top-level keys from extra (keep_alive, format, think, ...) pass straight through;
        # extra["options"] is merged over ours so a caller can override num_predict too.
        extra = dict(request.extra_for(self.name))
        extra_options = extra.pop("options", {})
        if not isinstance(extra_options, Mapping):
            raise TypeError(
                f"Request.extra['ollama']['options'] must be a mapping, "
                f"got {type(extra_options).__name__}"
            )
        options.update(extra_options)

        return {
            **extra,
            "model": model_id,
            "messages": messages,
            "stream": False,
            "options": options,
        }

    def _status_error(self, response: httpx.Response, model_id: str) -> ProviderError:
        try:
            detail = str(response.json()["error"])
        except (ValueError, KeyError, TypeError):
            detail = response.text[:200] or response.reason_phrase

        if response.status_code == httpx.codes.NOT_FOUND:
            return ProviderError(
                f"{detail} (try `ollama pull {model_id}`)", provider=self.name, model=model_id
            )
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            return RateLimited(
                detail,
                provider=self.name,
                model=model_id,
                retry_after=_retry_after(response.headers),
            )
        return ProviderError(
            f"HTTP {response.status_code}: {detail}", provider=self.name, model=model_id
        )


def _retry_after(headers: httpx.Headers) -> float | None:
    """Seconds from a ``Retry-After`` header. The HTTP-date form is ignored, not guessed."""
    try:
        return float(headers["retry-after"])
    except (KeyError, ValueError):
        return None

"""Anthropic, via the official ``anthropic`` SDK. Optional: ``pip install dumbwaiter[anthropic]``.

The SDK is imported when the provider is built, never at module scope, so the default
install neither needs it nor pays for importing it.

``anthropic`` 1.x runs on ``httpx2``, not ``httpx``. Anything handed to the SDK (a custom
``http_client``, a test transport) must come from ``httpx2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dumbwaiter.errors import ProviderError, ProviderTimeout, RateLimited
from dumbwaiter.providers.base import retry_after_seconds
from dumbwaiter.types import Completion, Request, Usage

if TYPE_CHECKING:
    import anthropic

_INSTALL_HINT = (
    "the Anthropic provider needs the 'anthropic' extra: pip install dumbwaiter[anthropic]"
)


def _import_sdk() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return anthropic


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self, client: anthropic.AsyncAnthropic | None = None, **client_kwargs: Any
    ) -> None:
        """Wrap ``client``, or build one from ``client_kwargs``.

        With neither, the SDK resolves credentials itself (``ANTHROPIC_API_KEY``, an
        ``ant auth login`` profile, ...). It also retries 429/5xx on its own before
        anything reaches us; pass ``max_retries=0`` to leave that to escalation policy.
        """
        self._sdk = _import_sdk()
        self._client = client or self._sdk.AsyncAnthropic(**client_kwargs)

    async def complete(self, request: Request, model_id: str) -> Completion:
        sdk = self._sdk
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system is not None:
            kwargs["system"] = request.system
        if request.stop:
            kwargs["stop_sequences"] = list(request.stop)
        # extra_body rather than kwargs: the 1.x SDK rejects unknown keyword arguments
        # (temperature among them), while extra_body is merged into the JSON as-is.
        extra = dict(request.extra_for(self.name))
        if extra:
            kwargs["extra_body"] = extra

        try:
            message = await self._client.messages.create(**kwargs)
        except sdk.APITimeoutError as exc:  # subclass of APIConnectionError: check first
            raise ProviderTimeout(str(exc), provider=self.name, model=model_id) from exc
        except sdk.RateLimitError as exc:
            raise RateLimited(
                exc.message,
                provider=self.name,
                model=model_id,
                retry_after=retry_after_seconds(exc.response.headers),
            ) from exc
        except sdk.APIStatusError as exc:
            raise ProviderError(
                f"HTTP {exc.status_code}: {exc.message}", provider=self.name, model=model_id
            ) from exc
        except sdk.APIError as exc:  # connection failures, unparseable responses
            raise ProviderError(str(exc), provider=self.name, model=model_id) from exc

        # Thinking and tool-use blocks carry no answer text; only text blocks do.
        text = "".join(block.text for block in message.content if block.type == "text")
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
            # Anthropic's vocabulary is the one Completion normalizes to.
            stop_reason=message.stop_reason,
            raw=message,
        )

    async def aclose(self) -> None:
        await self._client.close()

"""AWS Bedrock, via the Converse API. Optional: ``pip install dumbwaiter[bedrock]``.

boto3 is imported when the provider is built, never at module scope, and it is
synchronous, so every call runs in a worker thread rather than blocking the event loop.

Request and response shapes follow botocore's bundled ``bedrock-runtime`` service model.
Model ids, including regional inference-profile variants, must come from
``aws bedrock list-foundation-models`` against a real account.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from dumbwaiter.errors import ConfigError, ProviderError, ProviderTimeout, RateLimited
from dumbwaiter.providers.base import retry_after_seconds
from dumbwaiter.types import Completion, Request, Usage

_INSTALL_HINT = "the Bedrock provider needs the 'bedrock' extra: pip install dumbwaiter[bedrock]"


class BedrockProvider:
    name = "bedrock"

    def __init__(self, client: Any = None, **client_kwargs: Any) -> None:
        """Wrap a ``bedrock-runtime`` client, or build one from ``client_kwargs``.

        With neither, boto3 resolves region and credentials the usual way (environment,
        ``~/.aws/config``, instance role).
        """
        try:
            import boto3
            import botocore.exceptions
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        self._errors = botocore.exceptions
        if client is None:
            try:
                client = boto3.client("bedrock-runtime", **client_kwargs)
            except botocore.exceptions.BotoCoreError as exc:  # NoRegionError, mostly
                raise ConfigError(f"cannot build a Bedrock client: {exc}") from exc
        self._client = client

    async def complete(self, request: Request, model_id: str) -> Completion:
        kwargs = self._converse_kwargs(request, model_id)
        errors = self._errors
        try:
            response = await asyncio.to_thread(self._client.converse, **kwargs)
        except errors.ClientError as exc:
            raise self._client_error(exc, model_id) from exc
        except (errors.ReadTimeoutError, errors.ConnectTimeoutError) as exc:
            raise ProviderTimeout(str(exc), provider=self.name, model=model_id) from exc
        except errors.BotoCoreError as exc:  # no credentials, endpoint unreachable, ...
            raise ProviderError(str(exc), provider=self.name, model=model_id) from exc

        # Reasoning and tool-use blocks carry no answer text; only text blocks do.
        blocks = response["output"].get("message", {}).get("content", [])
        text = "".join(block["text"] for block in blocks if "text" in block)
        usage = response["usage"]
        return Completion(
            text=text,
            usage=Usage(input_tokens=usage["inputTokens"], output_tokens=usage["outputTokens"]),
            # Converse already uses end_turn / max_tokens / stop_sequence / tool_use; its
            # Bedrock-only reasons (guardrail_intervened, content_filtered) pass through.
            stop_reason=response["stopReason"],
            raw=response,
        )

    async def aclose(self) -> None:
        self._client.close()

    def _converse_kwargs(self, request: Request, model_id: str) -> dict[str, Any]:
        inference: dict[str, Any] = {"maxTokens": request.max_tokens}
        if request.stop:
            inference["stopSequences"] = list(request.stop)

        # Top-level Converse fields from extra (additionalModelRequestFields,
        # guardrailConfig, ...) pass through; extra["inferenceConfig"] is merged over ours.
        extra = dict(request.extra_for(self.name))
        extra_inference = extra.pop("inferenceConfig", {})
        if not isinstance(extra_inference, Mapping):
            raise TypeError(
                f"Request.extra['bedrock']['inferenceConfig'] must be a mapping, "
                f"got {type(extra_inference).__name__}"
            )
        inference.update(extra_inference)

        kwargs: dict[str, Any] = {
            **extra,
            "modelId": model_id,
            "messages": [
                {"role": m.role, "content": [{"text": m.content}]} for m in request.messages
            ],
            "inferenceConfig": inference,
        }
        if request.system is not None:
            kwargs["system"] = [{"text": request.system}]
        return kwargs

    def _client_error(self, exc: Any, model_id: str) -> ProviderError:
        error = exc.response.get("Error", {})
        code = error.get("Code", "Unknown")
        message = error.get("Message")
        detail = f"{code}: {message}" if message else code
        if code == "ThrottlingException":
            headers = exc.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
            return RateLimited(
                detail,
                provider=self.name,
                model=model_id,
                retry_after=retry_after_seconds(headers),
            )
        if code == "ModelTimeoutException":
            return ProviderTimeout(detail, provider=self.name, model=model_id)
        return ProviderError(detail, provider=self.name, model=model_id)

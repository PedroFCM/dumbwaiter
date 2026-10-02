"""Live calls against real providers. Run by hand; never part of the test suite.

uv run python scripts/smoke.py ollama/llama3.1:latest
uv run python scripts/smoke.py ollama/gemma4:latest "explain a mutex in one sentence" --no-think
"""

from __future__ import annotations

import argparse
import asyncio
import time

from dumbwaiter.providers import ProviderRegistry
from dumbwaiter.types import ModelRef, Request


async def main(spec: str, prompt: str, max_tokens: int, think: bool | None) -> None:
    ref = ModelRef.parse(spec)
    registry = ProviderRegistry()
    try:
        provider = registry.get(ref.provider)
        started = time.perf_counter()
        # Thinking models (gemma4, ...) spend max_tokens on reasoning first and can return
        # an empty answer; Ollama's `think` flag turns that off per request.
        extra = {} if think is None else {"ollama": {"think": think}}
        request = Request.user(prompt, max_tokens=max_tokens, extra=extra)
        completion = await provider.complete(request, ref.model_id)
        elapsed = time.perf_counter() - started
    finally:
        await registry.aclose()

    print(completion.text.strip() or "(empty answer)")
    if not completion.text.strip() and completion.stop_reason == "max_tokens":
        print("hint: budget spent before answering; try --no-think or more --max-tokens")
    print(
        f"\n[{ref}] {elapsed:.1f}s, stop={completion.stop_reason}, "
        f"in={completion.usage.input_tokens} out={completion.usage.output_tokens}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="provider/model-id, e.g. ollama/llama3.1:latest")
    parser.add_argument("prompt", nargs="?", default="Say hello in five words.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ollama thinking on/off; omitted leaves the model default",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model, args.prompt, args.max_tokens, args.think))

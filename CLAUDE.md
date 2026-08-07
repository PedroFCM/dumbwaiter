# CLAUDE.md — dumbwaiter

## What this is

A Python library that decides **which model should answer a given request**, then executes
it. Classify the prompt, pick the cheapest route that can handle it, call that provider.
Tagline: *most requests don't need the smart one.*

Providers: Ollama (local daemon, the default), AWS Bedrock (Converse API, via boto3), and
Anthropic (via the `anthropic` SDK).

**Local-first by default.** There is no AWS account, and Claude Pro grants no API access —
the Anthropic API is separately billed. So the default install runs entirely on the
machine: Ollama for completions, in-process ONNX embeddings for classification. Cloud
providers live behind extras and are written and stub-tested but not executable yet. Keep
it that way: anything that makes a cloud credential mandatory for the default path is a
regression.

## Status

| Milestone | State |
|---|---|
| M1 — types, config, validation, gates | done |
| M2 — providers (Bedrock + Anthropic), pricing | next |
| M3 — embedding classifier, `Router.route()` / `complete()` | |
| M4 — escalation policy, cost caps, streaming, other classifiers | |
| M5 — decision log, training from logs, eval harness | |
| M6 — docs, results table, OpenWebUI pipe example | |

## Layout and dependency direction

```
src/dumbwaiter/
  types.py          frozen dataclasses on the hot path
  config.py         pydantic, parsed from untrusted YAML
  errors.py         one exception root
  router.py         (M3) composes classifier + providers
  classifiers/      (M3+) Classifier protocol and strategies
  providers/        (M2) Provider protocol, ollama, bedrock, anthropic, registry
  embeddings/       (M3) EmbeddingBackend protocol, local (default), bedrock
  pricing.py        (M2) per-model prices with provenance
  observability.py  (M5) JSONL decision log
```

`core -> providers` and `core -> classifiers`. Nothing depends on `router`; the router
composes. Do not introduce an import from `providers/` or `classifiers/` back into
`types.py` or `config.py`.

## Rules that are easy to get wrong

**Never invent model ids, prices, or SDK versions.** Bedrock ids and their regional
inference-profile variants must come from `aws bedrock list-foundation-models` against a
real account, Ollama ids from `ollama list` on the machine, and prices from the provider's
pricing page. An unpriced model reports `cost_usd = None` — never a plausible-looking
guess. Every price row carries `source` and `checked_on` because partner (Bedrock) pricing
diverges from first-party and both drift.

**`None` and zero are different facts.** A local Ollama model costs a real `0`; an
unrecognised cloud model costs `None` because we do not know. Do not collapse them.

**Cloud SDKs are optional extras and must be imported lazily.** `boto3` and `anthropic`
are not core dependencies. Importing either at module scope breaks the default install;
import inside the provider and raise a message naming the extra to install.

**Do not add a generic `temperature` (or `top_p`, `top_k`, `budget_tokens`) to `Request`.**
Current Anthropic models reject all of them with a 400. The shared surface stays minimal —
messages, system, max_tokens, stop, tools, stream — and provider-specific parameters go in
`Request.extra`, merged only by the provider that owns them.

**Dependencies take lower bounds only, never upper caps.** A cap in a library becomes a
resolver conflict inside a consuming application. `uv.lock` is committed but governs this
repo's dev and CI only; installers ignore it. `anthropic` is the one deliberately tight
floor — it is pre-1.0 with no cross-minor compatibility promise.

**Do not add `python_version` to the mypy config.** Pinning it to the 3.11 floor makes
mypy parse numpy's 3.12+ stubs under 3.11 syntax rules and abort before checking anything.
Ruff's `target-version` enforces the floor on our own code instead.

**Keep the ruff pin and the pre-commit rev identical.** `ruff==0.16.1` in the dev group
must match `rev:` in `.pre-commit-config.yaml`, or the formatter fights itself between
local runs and hooks.

**Routes are named, not numbered.** `code` and `vision` are as valid as `simple` and
`complex`. The optional `tier` layers a cost ordering on top and must be set on every
route or on none — a partial ordering is not one.

## Conventions

- **Async-first.** Provider methods are `async`. boto3 is sync, so wrap it in
  `asyncio.to_thread` rather than blocking the loop.
- **Frozen dataclasses for hot-path values, pydantic for config.** Config is parsed from
  untrusted YAML and earns the validation; `Request`/`Decision` are built by us thousands
  of times and should stay cheap.
- **Validate at load, not at use.** A router that discovers a typo'd route name on the
  request that needed it is worse than useless.
- **Errors carry what escalation needs.** `ProviderError` holds provider and model;
  `RateLimited` holds `retry_after`. M4 branches on these, so map every provider SDK
  exception onto them rather than letting raw botocore or anthropic errors escape.

## Testing

- **No network in the test suite, ever.** Stub boto3 with `botocore.stub.Stubber` and the
  Anthropic SDK with `respx`. Live calls belong in `scripts/smoke.py`, run by hand.
- Branch coverage gated at 95% (`uv run pytest --cov`). It is at 100% today.
- Test behaviour that other code depends on, not structure. The error-hierarchy tests
  exist because escalation policy branches on those relationships.

## Commands

```bash
uv sync --all-extras
uv run pre-commit install     # once

uv run pytest
uv run pytest --cov
uv run mypy
uv run ruff check . && uv run ruff format --check .
```

## Git

- Branch off `main`; do not commit to it directly.
- **Conventional Commits.** Body explains *why*, not what the diff already shows.
- Open a PR; CI runs the same four gates.

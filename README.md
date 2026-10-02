# dumbwaiter

**Most requests don't need the smart one.**

A dumbwaiter is the small lift that carries things between floors. This one carries LLM
requests between model tiers: it classifies each request and sends it to the cheapest
route that can actually answer it.

**Local-first.** The default install runs entirely on your machine: [Ollama](https://ollama.com)
for completions and in-process ONNX embeddings for classification. No API key, no cloud
account. AWS Bedrock and Anthropic are optional extras for when you want a cloud rung on
the ladder.

> **Status: early.** Milestones 1 and 2 of 6 are done: configuration, plus providers that
> can call a model (Ollama runs today; Bedrock and Anthropic are written and stub-tested).
> The router that picks between them is next. See [Roadmap](#roadmap).

## The idea

Sending `"translate this to Portuguese"` to a frontier model is paying sledgehammer
prices to crack a nut. Sending `"design a rate limiter and justify the tradeoffs"` to a
small model gets you a confident, useless answer. A router picks per request.

What makes this one different from the routers that already exist:

| Project | Approach | Limitation this addresses |
|---|---|---|
| [RouteLLM](https://github.com/lm-sys/RouteLLM) | Trained routers + cost threshold | Binary strong/weak, pinned to the trained model pair |
| [semantic-router](https://github.com/aurelio-labs/semantic-router) | Embedding similarity over example utterances | Routes to intents, not a cost-tiered model ladder; no execution or cost accounting |
| [Arch-Router](https://huggingface.co/katanemo/Arch-Router-1.5B) | 1.5B model, domain/action preferences | No cost dimension; requires hosting a router model |
| [Bedrock Intelligent Prompt Routing](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html) | Managed, serverless | Two models, same family, Bedrock only, opaque decision |
| LiteLLM Router | Load balancing and fallbacks | Routes for availability, not task difficulty |

dumbwaiter is n-ary, crosses providers in a single ladder, lets you swap the
classification strategy, and hands back a `Decision` explaining itself.

## Configuration

```yaml
default_route: simple
min_confidence: 0.15

classifier:
  type: embedding          # or: rules, llm_judge, cascade, trained
  backend: local           # in-process ONNX; no network

routes:
  - name: simple
    model: ollama/llama3.1:latest
    tier: 0
    examples: ["what time is it in Lisbon", "summarize this paragraph", "translate this"]

  - name: complex
    model: ollama/gemma4:latest
    tier: 1
    examples: ["design a distributed rate limiter", "prove this invariant holds", "plan this migration"]

  # Optional cloud rung, behind the [anthropic] extra:
  # - name: frontier
  #   model: anthropic/claude-sonnet-5
  #   tier: 2

policy:
  escalate_on_error: true
  escalate_on_refusal: true
  max_cost_per_request_usd: 0.05   # bites once a priced cloud route is on the ladder
```

Model specs are `provider/model-id`. Ollama ids are whatever `ollama list` shows on your
machine.

Routes are **named**, not numbered — `code` and `vision` are as valid as `simple` and
`complex`. The optional `tier` adds an ordering on top, which is what makes cost caps and
escalation meaningful. Set it on every route or on none.

Everything checkable is checked at load: unknown route names, malformed model specs, tier
gaps, typo'd keys, and routes too thin on examples for the classifier you picked.

See [`examples/config.yaml`](examples/config.yaml) — the test suite loads it, so it cannot
drift out of sync with validation.

## Design notes

**The shared request surface is deliberately small.** Current Anthropic models reject
`temperature`, `top_p`, `top_k`, and `budget_tokens` with a 400. A normalized `Request`
carrying a generic `temperature` and forwarding it everywhere would be a latent bug, so
provider-specific parameters live in `Request.extra` and are merged only by the provider
that understands them.

**Prices carry provenance.** Every entry records its source and the date it was checked,
because partner (Bedrock) pricing diverges from first-party pricing and both drift. An
unpriced model reports `cost_usd = None` rather than a plausible-looking wrong number. A
local Ollama model costs a real `0`; that is a different fact from "unknown".

**Routing is separable from execution.** `route()` returns a `Decision` — chosen route,
per-route scores, confidence, and a human-readable reason — without spending anything.
`complete()` is `route()` plus a provider call.

## Roadmap

- [x] **M1** — types, config, validation, strict typing and lint gates
- [x] **M2** — providers behind one protocol: Ollama (default), plus Bedrock Converse and the Anthropic SDK as optional extras
- [ ] **M3** — embedding classifier (local ONNX by default) and the router itself
- [ ] **M4** — escalation policy, cost caps, streaming, rules / LLM-judge / cascade classifiers
- [ ] **M5** — decision log, training a classifier from your own traffic, eval harness
- [ ] **M6** — docs, results table, OpenWebUI pipe example

## Development

```bash
uv sync --all-extras          # include the cloud SDKs; their tests need them
uv run pre-commit install     # once

uv run pytest
uv run pytest --cov         # coverage, gated at 95%
uv run mypy
uv run ruff check . && uv run ruff format --check .
```

Requires Python 3.11+. The test suite makes no network calls and currently sits at 100%
branch coverage.

Commits run ruff (check + format), `uv-lock`, and a small set of file-hygiene hooks. The
`uv-lock` hook regenerates `uv.lock` whenever `pyproject.toml` changes, so the committed
lock can't go stale against the declared dependencies.

### Dependency policy

Runtime dependencies carry **lower bounds only — no upper caps**. This is a library, and
a cap here turns into a resolver conflict inside somebody else's application.

`uv.lock` is committed, but it constrains only this repo's development and CI. Installing
`dumbwaiter` as a dependency ignores it entirely — you get whatever your own resolver
picks within the bounds above.

The one deliberately tight floor is `anthropic`: it is pre-1.0 and makes no compatibility
promises across minor releases, so claiming support for old versions we have never run
would be a false promise.

## License

MIT

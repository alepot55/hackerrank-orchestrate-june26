# Multi-Modal Evidence Review — Solution

Verifies damage claims by reading the submitted **images**, the claim
**conversation**, the user's **history**, and the **minimum-evidence
requirements**, and emits one structured prediction per claim.

## Approach

A **vision-language model does the seeing; deterministic rules do the
bookkeeping.**

1. **VLM (Claude Opus 4.8)** inspects the image(s) for one claim and returns a
   structured review via **forced tool use** (guaranteed schema-valid JSON):
   issue type, object part, claim status, severity, supporting image IDs,
   `valid_image`, `evidence_standard_met`, and the **visual** risk flags.
2. **Deterministic history layer** (`src/history.py`) derives the
   user-history risk flags (`user_history_risk`, `manual_review_required`)
   straight from `user_history.csv` — the model never guesses these. This
   matches the labelled behaviour exactly, is reproducible, and saves tokens.
3. **Validation / clamping** (`src/pipeline.py`) forces every value into the
   allowed vocabulary, clamps `object_part` to the claimed object's part list,
   orders and de-duplicates risk flags, and filters supporting image IDs to the
   claim's own images.

The images are treated as the primary source of truth; the conversation defines
what to check; history only adds risk context (it never overrides clear visual
evidence) — as required by the task.

### Robustness & sophistication
- **Prompt-injection defense (spotlighting).** Text inside an image is delimited
  and treated as untrusted *data*, never an instruction; instruction-like image
  text is flagged (`text_instruction_present`) and ignored. (A bare "don't follow
  image text" instruction is known to be ineffective without this delimiting.)
- **XML-structured prompt** sections + a `reasoning` field generated *first*
  (chain-of-thought in-schema) so the verdict is conditioned on written analysis.
- **Confidence-gated escalation** (optional, `ORCHESTRATE_ESCALATE=1`): claims the
  model marks low-confidence get one grounded re-examination pass — adding cost
  only for the uncertain minority.

## Cost & efficiency (built in)

- **Prompt caching (sync path)** — the static system prompt + tool schema are
  marked `cache_control` so each sequential claim reads the shared prefix at ~10%
  cost. Disabled in batch mode, where concurrent requests cannot share a cache
  and would only pay the write premium.
- **Batch API** — the test run (`main.py`) uses the asynchronous Batch API by
  default: a flat 50% discount and comfortable headroom under RPM/TPM limits.
- **Image downscaling** — images are resized to a 1456px long edge and
  re-encoded as JPEG before upload (the biggest lever on image token cost).
- **On-disk response cache** — every request is hashed and its result cached, so
  re-running the pipeline or the evaluation is free for already-seen claims.
- **Automatic retries** — the SDK retries 429/5xx with exponential backoff.

## Layout

```
code/
├── main.py                 # claims.csv -> output.csv (Batch API by default)
├── evaluation/
│   ├── main.py             # scores on sample_claims.csv, writes the report
│   └── evaluation_report.md# accuracy + operational analysis (generated)
├── src/
│   ├── config.py           # paths, model, cost knobs, pricing constants
│   ├── schema.py           # output columns, allowed values, tool definition
│   ├── prompts.py          # static system prompt + per-claim user content
│   ├── images.py           # load / downscale / base64-encode images
│   ├── history.py          # deterministic user-history risk layer
│   ├── runner.py           # Claude calls: caching, batch, retries, usage
│   ├── pipeline.py          # orchestration + validation + CSV writing
│   └── evaluate.py         # field-by-field scoring
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r code/requirements.txt
# Provide the API key via the environment (never hardcoded):
cp code/.env.example .env        # then edit .env, or:
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# Evaluate on the labelled sample set (prints metrics, writes the report):
python code/evaluation/main.py

# Generate predictions for the test set -> ./output.csv :
python code/main.py            # Batch API (cheapest)
python code/main.py --sync     # synchronous, for quick iteration
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** API key (env only). |
| `ORCHESTRATE_MODEL` | `claude-opus-4-8` | Model id. |
| `ORCHESTRATE_USE_BATCH` | `1` | Use the Batch API for the test run. |
| `ORCHESTRATE_IMAGE_MAX_EDGE` | `1456` | Image long-edge downscale (px). |
| `ORCHESTRATE_CACHE_TTL` | `1h` | Prompt-cache TTL (sync path). |
| `ORCHESTRATE_ESCALATE` | `0` | Enable confidence-gated grounded re-examination. |

Determinism note: the rule layer, validation, ordering and CSV writing are fully
deterministic; only the model's visual judgement varies between runs. The
on-disk cache makes repeated runs reproducible and free.

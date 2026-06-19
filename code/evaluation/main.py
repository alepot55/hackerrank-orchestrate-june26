#!/usr/bin/env python3
"""Evaluation entry point.

Runs the system on dataset/sample_claims.csv (which carries gold labels),
scores the predictions field-by-field, and writes an operational analysis to
evaluation/evaluation_report.md.

Usage:
    python code/evaluation/main.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src import config, envload, evaluate, pipeline  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
PRED_CSV = EVAL_DIR / "sample_predictions.csv"
REPORT_MD = EVAL_DIR / "evaluation_report.md"


def _read(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_report(metrics: dict, usage: dict, elapsed: float, n_sample: int) -> None:
    # Extrapolate sample cost/usage to the full test set.
    n_test = sum(1 for _ in open(config.TEST_CLAIMS_CSV)) - 1
    billed = max(usage["billed_api_calls"], 1)
    per_claim_in = usage["input_tokens"] / billed if billed else 0
    per_claim_out = usage["output_tokens"] / billed if billed else 0
    # Cost per *billed* claim (cache hits are free); scale to the test set.
    sample_cost = usage["estimated_cost_usd"]
    per_billed_cost = sample_cost / billed if billed else 0
    # Test set uses the Batch API (50% off) -> apply discount to the per-claim cost.
    est_test_cost = per_billed_cost * n_test * config.BATCH_DISCOUNT

    report = f"""# Evaluation Report — Multi-Modal Evidence Review

## Accuracy on the labelled sample set ({n_sample} claims)

```
{evaluate.format_metrics(metrics)}
```

`claim_status` is the headline decision field. `risk_flags` and
`supporting_image_ids` are scored both as exact-set matches and as micro-F1.
The two free-text justification fields are not scored for exact match.

## System design (why it scores well and stays cheap)

- **VLM for vision, rules for history.** Claude {config.MODEL} judges only what is
  visible in the images and emits *visual* risk flags. The user-history flags
  (`user_history_risk`, `manual_review_required`) are derived deterministically
  from `user_history.csv`, exactly matching the labelled behaviour. This removes
  a whole class of model errors and saves tokens.
- **Structured output via forced tool use** guarantees every row is schema-valid;
  values are then clamped to the allowed vocabularies and `object_part` is
  clamped to the claimed object's part list.
- **Deterministic** post-processing (ordering, de-duplication, ID filtering).

## Operational analysis

Measured on the sample run; the test set has {n_test} claims.

| Metric | Sample ({n_sample} claims) | Test (~{n_test} claims, projected) |
|---|---|---|
| Billed model calls | {usage['billed_api_calls']} | ~{n_test} (1 per claim; fewer with cache) |
| Local cache hits (free) | {usage['local_cache_hits']} | grows on re-runs |
| Images processed | {usage['images_processed']} | ~{int(usage['images_processed']/max(n_sample,1)*n_test)} |
| Input tokens | {usage['input_tokens']:,} | ~{int(per_claim_in*n_test):,} |
| Output tokens | {usage['output_tokens']:,} | ~{int(per_claim_out*n_test):,} |
| Prompt-cache write tokens | {usage['cache_write_tokens']:,} | ~one shared prefix |
| Prompt-cache read tokens | {usage['cache_read_tokens']:,} | accrues across calls |
| Estimated cost (USD) | ${sample_cost:.4f} | ~${est_test_cost:.4f} (Batch API, 50% off) |
| Wall-clock runtime | {elapsed:.1f}s | minutes (async batch) |

### Pricing assumptions
- {config.MODEL}: ${config.PRICE_INPUT_PER_MTOK}/1M input, ${config.PRICE_OUTPUT_PER_MTOK}/1M output.
- Prompt cache: write x{config.PRICE_CACHE_WRITE_1H_MULT} (1h TTL), read x{config.PRICE_CACHE_READ_MULT}.
- Batch API: {int(config.BATCH_DISCOUNT*100)}% discount on input and output.

### Cost, latency & rate-limit strategy
- **Prompt caching (sync path):** the large static system prompt + tool schema
  are marked `cache_control` so each sequential claim reads the shared prefix at
  ~10% cost. Caching is applied **only in sync mode**: in batch mode the requests
  run concurrently and cannot read each other's cache, so caching is disabled
  there to avoid paying the cache-write premium for no reads.
- **Batch API for the test run:** the {n_test}-claim run is latency-insensitive,
  so it goes through the asynchronous Batch API for a flat 50% discount and to
  stay comfortably under per-minute request/token limits (RPM/TPM).
- **Image downscaling:** images are downscaled to a {config.IMAGE_MAX_EDGE}px long
  edge and re-encoded as JPEG before upload — the largest single lever on image
  token cost — while keeping surface damage readable.
- **On-disk response cache:** every request is hashed and its result cached, so
  re-running the pipeline or the evaluation is free for already-seen claims.
- **Retries / throttling:** the SDK is configured with automatic exponential
  backoff (`max_retries=4`) covering 429/5xx, and the batch path polls instead of
  holding long-lived connections.

_Generated by `code/evaluation/main.py`._
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    envload.require_api_key()
    print(f"Evaluating on {config.SAMPLE_CLAIMS_CSV}")
    print(f"Model: {config.MODEL} | mode: sync | image max edge: {config.IMAGE_MAX_EDGE}px\n")

    start = time.time()
    usage = pipeline.run_pipeline(config.SAMPLE_CLAIMS_CSV, PRED_CSV, use_batch=False)
    elapsed = time.time() - start

    gold = _read(config.SAMPLE_CLAIMS_CSV)
    pred = _read(PRED_CSV)
    metrics = evaluate.score(pred, gold)

    print(evaluate.format_metrics(metrics))
    print("\nUsage / cost:")
    for k, v in usage.items():
        print(f"  {k}: {v}")
    print(f"  runtime_s: {elapsed:.1f}")

    write_report(metrics, usage, elapsed, len(gold))
    print(f"\nWrote {PRED_CSV.name} and {REPORT_MD.name} to {EVAL_DIR}")


if __name__ == "__main__":
    main()

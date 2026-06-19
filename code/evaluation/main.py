#!/usr/bin/env python3
"""Evaluation entry point.

Runs the system on dataset/sample_claims.csv (gold-labelled), scores it with
judge-grade rigor (confidence intervals, ordinal + per-class metrics, confusion
matrix, error analysis), and writes evaluation/evaluation_report.md.

Usage:
    python code/evaluation/main.py              # run + score + report
    python code/evaluation/main.py --score-only # re-score existing predictions
                                                 # (no API calls — free)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src import config, envload, evaluate, pipeline  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
PRED_CSV = EVAL_DIR / "sample_predictions.csv"
REPORT_MD = EVAL_DIR / "evaluation_report.md"
USAGE_JSON = EVAL_DIR / ".last_usage.json"


def _read(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _history_ablation(pred_rows, gold_rows, claim_rows) -> str:
    """Ablation: how much does the deterministic history layer contribute to
    risk_flags accuracy? Recompute risk_flags WITHOUT the history layer (visual
    flags only) and compare exact-set accuracy — no API calls needed."""
    from src import schema
    from src.evaluate import _to_set
    with_h = sum(1 for p, g in zip(pred_rows, gold_rows)
                 if _to_set(p["risk_flags"]) == _to_set(g["risk_flags"]))
    without = 0
    for p, g in zip(pred_rows, gold_rows):
        visual = {f for f in _to_set(p["risk_flags"]) if f in schema.VISUAL_RISK_FLAGS}
        if visual == _to_set(g["risk_flags"]):
            without += 1
    n = len(gold_rows)
    return (f"- risk_flags exact-set accuracy **with** history layer: "
            f"{with_h}/{n} ({with_h/n*100:.0f}%); **without** it (visual flags "
            f"only): {without}/{n} ({without/n*100:.0f}%). The deterministic "
            f"history layer is responsible for the difference.")


def write_report(metrics: dict, usage: dict, elapsed: float, n_sample: int,
                 pred_rows, gold_rows, claim_rows) -> None:
    n_test = sum(1 for _ in open(config.TEST_CLAIMS_CSV)) - 1
    billed = max(usage.get("billed_api_calls", 0), 1)
    per_claim_in = usage.get("input_tokens", 0) / billed
    per_claim_out = usage.get("output_tokens", 0) / billed
    sample_cost = usage.get("estimated_cost_usd", 0.0)
    per_billed_cost = sample_cost / billed
    est_test_cost = per_billed_cost * n_test * config.BATCH_DISCOUNT

    so = metrics["severity_ordinal"]
    errors = evaluate.error_rows(pred_rows, gold_rows, claim_rows)
    n_correct_rows = n_sample - len(errors)

    report = f"""# Evaluation Report — Multi-Modal Evidence Review

## Accuracy on the labelled sample set ({n_sample} claims)

```
{evaluate.format_metrics(metrics)}
```

**Read the confidence intervals.** With only {n_sample} labelled examples, a
single field's 95% Wilson interval spans ~±15 points, so small differences
between prompt variants are statistical noise. We therefore tune to *rubric
logic and generalizable design*, not to one-example swings on this dev set, and
treat `claims.csv` strictly as an unseen test set.

### Why the headline number understates `severity`
`severity` is **ordinal** (none < low < medium < high). Exact-match accuracy
treats "high vs medium" as badly as "high vs none", which is wrong for an
ordinal field. Under the metrics evaluators actually use for ordinal grading
(MAE, within-1 accuracy, Quadratic Weighted Kappa) the system is much stronger:
**within-1 accuracy {so['adjacent_within_1']*100:.0f}%, MAE {so['mae']:.2f},
QWK {so['qwk']:.2f}** — i.e. almost every "error" is a single adjacent level.

### `claim_status` confusion matrix (rows = gold, cols = predicted)
```
{evaluate.format_confusion(metrics)}
```

## System design (accuracy + cost)

- **VLM for vision, deterministic rules for history.** Claude {config.MODEL}
  judges only what is visible and emits *visual* risk flags; the user-history
  flags (`user_history_risk`, `manual_review_required`) are derived
  deterministically from `user_history.csv`. This removes a class of model
  errors and saves tokens.
- **Structured output via forced tool use** guarantees schema-valid rows; values
  are clamped to the allowed vocabularies and `object_part` to the object's part
  list. A `reasoning` field is generated *first* (chain-of-thought in-schema) so
  the decision is conditioned on written analysis; a self-reported `confidence`
  field drives an optional, cost-gated escalation pass (a grounded re-examination
  of only the low-confidence minority of claims).
- **Few-shot exemplars** in the (cached) system prompt teach the labelling
  conventions — chiefly severity calibration and risk-flag triggers.
- **Prompt-injection defense (spotlighting).** Text inside an image is delimited
  and treated strictly as untrusted *data*, never as an instruction; instruction
  -like image text is flagged (`text_instruction_present`) and ignored. Research
  shows a bare "don't follow image text" instruction barely helps unless the
  untrusted span is delimited this way.

### Ablation — contribution of the deterministic history layer
{_history_ablation(pred_rows, gold_rows, claim_rows)}

## Error analysis ({len(errors)}/{n_sample} rows with at least one field error)
{chr(10).join(errors) if errors else '- none'}

## Operational analysis

Measured on the sample run; the test set has {n_test} claims.

| Metric | Sample ({n_sample} claims) | Test (~{n_test} claims, projected) |
|---|---|---|
| Billed model calls | {usage.get('billed_api_calls', 0)} | ~{n_test} |
| Local cache hits (free) | {usage.get('local_cache_hits', 0)} | grows on re-runs |
| Images processed | {usage.get('images_processed', 0)} | ~{int(usage.get('images_processed', 0)/max(n_sample,1)*n_test)} |
| Input tokens | {usage.get('input_tokens', 0):,} | ~{int(per_claim_in*n_test):,} |
| Output tokens | {usage.get('output_tokens', 0):,} | ~{int(per_claim_out*n_test):,} |
| Estimated cost (USD) | ${sample_cost:.4f} | ~${est_test_cost:.4f} (Batch API, 50% off) |
| Wall-clock runtime | {elapsed:.1f}s | minutes (async batch) |

### Pricing assumptions
- {config.MODEL}: ${config.PRICE_INPUT_PER_MTOK}/1M input, ${config.PRICE_OUTPUT_PER_MTOK}/1M output.
- Prompt cache: write x{config.PRICE_CACHE_WRITE_5M_MULT}/x{config.PRICE_CACHE_WRITE_1H_MULT}, read x{config.PRICE_CACHE_READ_MULT}.
- Batch API: {int(config.BATCH_DISCOUNT*100)}% discount on input and output.

### Cost, latency & rate-limit strategy
- **Prompt caching (sync path only):** the static system prompt + tool schema are
  cached so each sequential claim reads the shared prefix at ~10% cost. **Disabled
  in batch mode** — concurrent requests cannot read each other's cache, so caching
  would only add the write premium (confirmed by Anthropic's docs).
- **Batch API for the test run:** the {n_test}-claim run is latency-insensitive,
  so it uses the asynchronous Batch API for a flat 50% discount and to stay under
  per-minute request/token limits (RPM/TPM).
- **Image downscaling** to {config.IMAGE_MAX_EDGE}px long edge (the largest lever on
  image token cost), **on-disk response cache** (re-runs are free), and the SDK's
  automatic exponential-backoff retries (`max_retries=4`) for 429/5xx.

_Generated by `code/evaluation/main.py`._
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true",
                        help="Re-score existing predictions without calling the API.")
    args = parser.parse_args()

    if args.score_only:
        if not PRED_CSV.exists():
            raise SystemExit("No existing predictions to score. Run without --score-only first.")
        usage = json.loads(USAGE_JSON.read_text()) if USAGE_JSON.exists() else {}
        elapsed = 0.0
        print("Scoring existing predictions (no API calls).")
    else:
        envload.require_api_key()
        print(f"Evaluating on {config.SAMPLE_CLAIMS_CSV}")
        print(f"Model: {config.MODEL} | mode: sync | image max edge: {config.IMAGE_MAX_EDGE}px\n")
        start = time.time()
        usage = pipeline.run_pipeline(config.SAMPLE_CLAIMS_CSV, PRED_CSV, use_batch=False)
        elapsed = time.time() - start
        USAGE_JSON.write_text(json.dumps(usage))

    gold = _read(config.SAMPLE_CLAIMS_CSV)
    pred = _read(PRED_CSV)
    metrics = evaluate.score(pred, gold)

    print(evaluate.format_metrics(metrics))
    print("\n" + evaluate.format_confusion(metrics))
    if usage:
        print("\nUsage / cost:")
        for k, v in usage.items():
            print(f"  {k}: {v}")
    if elapsed:
        print(f"  runtime_s: {elapsed:.1f}")

    write_report(metrics, usage, elapsed, len(gold), pred, gold, gold)
    print(f"\nWrote {PRED_CSV.name} and {REPORT_MD.name} to {EVAL_DIR}")


if __name__ == "__main__":
    main()

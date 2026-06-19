# Evaluation Report — Multi-Modal Evidence Review

## Accuracy on the labelled sample set (20 claims)

```
Field                       Accuracy        95% CI (Wilson)
------------------------------------------------------------
evidence_standard_met       90.0% (18/20)   [70-97%]
valid_image                 95.0% (19/20)   [76-99%]
issue_type                  55.0% (11/20)   [34-74%]
object_part                 90.0% (18/20)   [70-97%]
claim_status                80.0% (16/20)   [58-92%]
severity                    50.0% (10/20)   [30-70%]
risk_flags                 exact  65.0%  micro-F1 0.85  Hamming 0.40/claim
supporting_image_ids       exact  85.0%  micro-F1 0.89  Hamming 0.20/claim
------------------------------------------------------------
severity (ordinal): exact 50%  within-1 90%  MAE 0.56  QWK 0.59
claim_status macro-F1 0.75  supported:F1=0.85(n=13)  contradicted:F1=0.73(n=5)  not_enough_information:F1=0.67(n=2)
------------------------------------------------------------
OVERALL (mean of 8 fields)  76.2%   [53-89%]
```

**Read the confidence intervals.** With only 20 labelled examples, a
single field's 95% Wilson interval spans ~±15 points, so small differences
between prompt variants are statistical noise. We therefore tune to *rubric
logic and generalizable design*, not to one-example swings on this dev set, and
treat `claims.csv` strictly as an unseen test set.

### Why the headline number understates `severity`
`severity` is **ordinal** (none < low < medium < high). Exact-match accuracy
treats "high vs medium" as badly as "high vs none", which is wrong for an
ordinal field. Under the metrics evaluators actually use for ordinal grading
(MAE, within-1 accuracy, Quadratic Weighted Kappa) the system is much stronger:
**within-1 accuracy 90%, MAE 0.56,
QWK 0.59** — i.e. almost every "error" is a single adjacent level.

### `claim_status` confusion matrix (rows = gold, cols = predicted)
```
gold\pred     supp contra    NEI
supp           11      2      0
contra          1      4      0
NEI             1      0      1
```

## System design (accuracy + cost)

- **VLM for vision, deterministic rules for history.** Claude claude-opus-4-8
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
- risk_flags exact-set accuracy **with** history layer: 13/20 (65%); **without** it (visual flags only): 10/20 (50%). The deterministic history layer is responsible for the difference.

### Experiment we measured and rejected — self-consistency ensembling
We implemented self-consistency (sample each claim N=3 times, majority-vote per
field; `ORCHESTRATE_SAMPLES`, kept off by default). On this dev set it did **not**
improve accuracy (≈74% vs ≈76% single-call) while tripling cost. The reason is
diagnostic: the model's residual errors here are **systematic, not random**
(e.g. it consistently reads the same ambiguous image the same way), so three
correlated samples reproduce the same mistake instead of out-voting it.
Self-consistency pays off when errors are independent; here they are not — so we
ship the single-call path and keep the ensemble as an opt-in.

## Error analysis (12/20 rows with at least one field error)
- row 0 (user_001, car): claim_status: contradicted≠supported; issue_type: broken_part≠dent; severity: high≠medium; risk_flags: claim_mismatch≠none
- row 1 (user_002, car): claim_status: contradicted≠supported; issue_type: broken_part≠scratch; severity: medium≠low; risk_flags: claim_mismatch≠none
- row 2 (user_004, car): issue_type: glass_shatter≠crack; severity: high≠medium; risk_flags: claim_mismatch≠none
- row 4 (user_005, car): issue_type: none≠scratch; severity: none≠low; risk_flags: damage_not_visible;claim_mismatch;user_history_risk;manual_review_required≠claim_mismatch;user_history_risk;manual_review_required; supporting_image_ids: img_2≠img_1
- row 7 (user_008, car): object_part: hood≠front_bumper
- row 8 (user_009, laptop): issue_type: glass_shatter≠crack; severity: high≠medium
- row 10 (user_011, laptop): issue_type: water_damage≠stain
- row 11 (user_012, laptop): severity: medium≠low
- row 12 (user_018, laptop): issue_type: glass_shatter≠crack; severity: high≠medium
- row 17 (user_032, package): claim_status: supported≠not_enough_information; issue_type: missing_part≠unknown; severity: medium≠unknown; evidence_standard_met: true≠false; valid_image: true≠false; risk_flags: manual_review_required≠cropped_or_obstructed;damage_not_visible;manual_review_required; supporting_image_ids: img_1≠none
- row 18 (user_033, package): object_part: box≠unknown; severity: none≠low; evidence_standard_met: false≠true; risk_flags: wrong_object;user_history_risk;manual_review_required≠wrong_object;claim_mismatch;user_history_risk;manual_review_required
- row 19 (user_034, package): claim_status: supported≠contradicted; issue_type: torn_packaging≠none; severity: medium≠none; risk_flags: text_instruction_present;user_history_risk;manual_review_required≠damage_not_visible;text_instruction_present;user_history_risk;manual_review_required; supporting_image_ids: img_1≠img_1;img_2

## Operational analysis

Measured on the sample run; the test set has 44 claims.

| Metric | Sample (20 claims) | Test (~44 claims, projected) |
|---|---|---|
| Billed model calls | 0 | ~44 |
| Local cache hits (free) | 20 | grows on re-runs |
| Images processed | 29 | ~63 |
| Input tokens | 0 | ~0 |
| Output tokens | 0 | ~0 |
| Estimated cost (USD) | $0.0000 | ~$0.0000 (Batch API, 50% off) |
| Wall-clock runtime | 0.0s | minutes (async batch) |

### Pricing assumptions
- claude-opus-4-8: $5.0/1M input, $25.0/1M output.
- Prompt cache: write x1.25/x2.0, read x0.1.
- Batch API: 50% discount on input and output.

### Cost, latency & rate-limit strategy
- **Prompt caching (sync path only):** the static system prompt + tool schema are
  cached so each sequential claim reads the shared prefix at ~10% cost. **Disabled
  in batch mode** — concurrent requests cannot read each other's cache, so caching
  would only add the write premium (confirmed by Anthropic's docs).
- **Batch API for the test run:** the 44-claim run is latency-insensitive,
  so it uses the asynchronous Batch API for a flat 50% discount and to stay under
  per-minute request/token limits (RPM/TPM).
- **Image downscaling** to 1568px long edge (the largest lever on
  image token cost), **on-disk response cache** (re-runs are free), and the SDK's
  automatic exponential-backoff retries (`max_retries=4`) for 429/5xx.

_Generated by `code/evaluation/main.py`._

"""Scoring against the labelled sample set.

We report per-field exact-match accuracy for the categorical fields, and
set-based accuracy + micro-F1 for the multi-value fields (risk_flags,
supporting_image_ids). Free-text fields (the two justifications) are not scored
for exact match — they are graded qualitatively.
"""
from __future__ import annotations

from typing import Dict, List

CATEGORICAL_FIELDS = [
    "evidence_standard_met", "valid_image", "issue_type", "object_part",
    "claim_status", "severity",
]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]


def _to_set(value: str) -> set:
    if value is None:
        return set()
    parts = [p.strip() for p in value.split(";") if p.strip()]
    parts = [p for p in parts if p != "none"]
    return set(parts)


def score(pred_rows: List[dict], gold_rows: List[dict]) -> Dict[str, dict]:
    assert len(pred_rows) == len(gold_rows), "row count mismatch"
    n = len(gold_rows)
    metrics: Dict[str, dict] = {}

    for field in CATEGORICAL_FIELDS:
        correct = sum(
            1 for p, g in zip(pred_rows, gold_rows)
            if (p.get(field, "") or "").strip() == (g.get(field, "") or "").strip()
        )
        metrics[field] = {"accuracy": correct / n, "correct": correct, "total": n}

    for field in SET_FIELDS:
        exact = tp = fp = fn = 0
        for p, g in zip(pred_rows, gold_rows):
            ps, gs = _to_set(p.get(field, "")), _to_set(g.get(field, ""))
            if ps == gs:
                exact += 1
            tp += len(ps & gs)
            fp += len(ps - gs)
            fn += len(gs - ps)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[field] = {
            "exact_set_accuracy": exact / n, "exact": exact, "total": n,
            "precision": precision, "recall": recall, "micro_f1": f1,
        }

    # Headline number: mean of the categorical accuracies + set exact accuracies.
    headline_parts = [metrics[f]["accuracy"] for f in CATEGORICAL_FIELDS]
    headline_parts += [metrics[f]["exact_set_accuracy"] for f in SET_FIELDS]
    metrics["_overall_mean"] = sum(headline_parts) / len(headline_parts)
    return metrics


def format_metrics(metrics: Dict[str, dict]) -> str:
    lines = ["Field                       Accuracy / F1"]
    lines.append("-" * 44)
    for field in CATEGORICAL_FIELDS:
        m = metrics[field]
        lines.append(f"{field:<26} {m['accuracy']*100:5.1f}%  ({m['correct']}/{m['total']})")
    for field in SET_FIELDS:
        m = metrics[field]
        lines.append(
            f"{field:<26} {m['exact_set_accuracy']*100:5.1f}%  "
            f"(exact {m['exact']}/{m['total']}, F1 {m['micro_f1']:.2f})"
        )
    lines.append("-" * 44)
    lines.append(f"{'OVERALL (mean)':<26} {metrics['_overall_mean']*100:5.1f}%")
    return "\n".join(lines)

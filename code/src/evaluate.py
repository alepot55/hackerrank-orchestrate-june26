"""Scoring against the labelled sample set — with the rigor judges reward.

Beyond plain accuracy we report:
  * Wilson 95% confidence intervals (a 20-example accuracy is very noisy);
  * ordinal metrics for `severity` (exact, off-by-one/adjacent accuracy, MAE,
    and Quadratic Weighted Kappa) — because severity is ordinal and an
    adjacent error (high vs medium) is far less wrong than a far error;
  * per-class precision/recall/F1 and a confusion matrix for `claim_status`;
  * micro/macro-F1, Hamming loss and exact-set accuracy for the multi-label
    risk_flags / supporting_image_ids fields;
  * an automated error listing for error analysis.
"""
from __future__ import annotations

import math
from typing import Dict, List

from . import schema

CATEGORICAL_FIELDS = [
    "evidence_standard_met", "valid_image", "issue_type", "object_part",
    "claim_status", "severity",
]
SET_FIELDS = ["risk_flags", "supporting_image_ids"]
SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": -1}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_set(value: str) -> set:
    if value is None:
        return set()
    parts = [p.strip() for p in value.split(";") if p.strip()]
    return set(p for p in parts if p != "none")


def wilson_ci(correct: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _fields(rows, field):
    return [(r.get(field, "") or "").strip() for r in rows]


# ---------------------------------------------------------------------------
# ordinal metrics (severity)
# ---------------------------------------------------------------------------
def ordinal_metrics(pred_rows, gold_rows, field="severity") -> dict:
    p = _fields(pred_rows, field)
    g = _fields(gold_rows, field)
    n = len(g)
    exact = 0
    abs_err = 0.0
    comparable = 0
    pv, gv = [], []
    for pi, gi in zip(p, g):
        if pi == gi:
            exact += 1
        po, go = SEVERITY_ORDER.get(pi, -1), SEVERITY_ORDER.get(gi, -1)
        if po >= 0 and go >= 0:  # both on the ordinal scale (exclude 'unknown')
            comparable += 1
            abs_err += abs(po - go)
            pv.append(po)
            gv.append(go)
    return {
        "exact": exact / n if n else 0.0,
        "adjacent_within_1": _adjacent_rate(p, g),
        "mae": abs_err / comparable if comparable else 0.0,
        "qwk": _qwk(pv, gv),
        "n": n,
    }


def _adjacent_rate(p, g) -> float:
    n = len(g)
    ok = 0
    for pi, gi in zip(p, g):
        if pi == gi:
            ok += 1
            continue
        po, go = SEVERITY_ORDER.get(pi, -99), SEVERITY_ORDER.get(gi, 99)
        if po >= 0 and go >= 0 and abs(po - go) <= 1:
            ok += 1
    return ok / n if n else 0.0


def _qwk(pv, gv, k=4) -> float:
    """Quadratic Weighted Kappa over the 0..3 severity scale (excludes unknown)."""
    if not gv:
        return 0.0
    O = [[0] * k for _ in range(k)]
    for a, b in zip(pv, gv):
        O[a][b] += 1
    n = len(gv)
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            e = row[i] * col[j] / n
            num += w * O[i][j]
            den += w * e
    return 1 - num / den if den else 1.0


# ---------------------------------------------------------------------------
# per-class P/R/F1 and confusion matrix
# ---------------------------------------------------------------------------
def per_class_prf(pred_rows, gold_rows, field, labels) -> dict:
    p = _fields(pred_rows, field)
    g = _fields(gold_rows, field)
    out = {}
    for lab in labels:
        tp = sum(1 for pi, gi in zip(p, g) if pi == lab and gi == lab)
        fp = sum(1 for pi, gi in zip(p, g) if pi == lab and gi != lab)
        fn = sum(1 for pi, gi in zip(p, g) if pi != lab and gi == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for gi in g if gi == lab)
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
    macro_f1 = sum(v["f1"] for v in out.values()) / len(labels) if labels else 0.0
    return {"per_class": out, "macro_f1": macro_f1}


def confusion(pred_rows, gold_rows, field, labels) -> List[List[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for pi, gi in zip(_fields(pred_rows, field), _fields(gold_rows, field)):
        if pi in idx and gi in idx:
            m[idx[gi]][idx[pi]] += 1  # rows=gold, cols=pred
    return m


# ---------------------------------------------------------------------------
# multi-label metrics
# ---------------------------------------------------------------------------
def multilabel(pred_rows, gold_rows, field) -> dict:
    exact = tp = fp = fn = 0
    hamming = 0
    universe = set(schema.ALL_RISK_FLAGS) - {"none"}
    n = len(gold_rows)
    for p, g in zip(pred_rows, gold_rows):
        ps, gs = _to_set(p.get(field, "")), _to_set(g.get(field, ""))
        if ps == gs:
            exact += 1
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
        hamming += len(ps ^ gs)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "exact_set_accuracy": exact / n if n else 0.0, "exact": exact, "total": n,
        "precision": precision, "recall": recall, "micro_f1": micro_f1,
        "hamming_per_claim": hamming / n if n else 0.0,
    }


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------
def score(pred_rows: List[dict], gold_rows: List[dict]) -> Dict[str, dict]:
    assert len(pred_rows) == len(gold_rows), "row count mismatch"
    n = len(gold_rows)
    metrics: Dict[str, dict] = {}
    for field in CATEGORICAL_FIELDS:
        correct = sum(1 for p, g in zip(pred_rows, gold_rows)
                      if (p.get(field, "") or "").strip() == (g.get(field, "") or "").strip())
        lo, hi = wilson_ci(correct, n)
        metrics[field] = {"accuracy": correct / n if n else 0.0, "correct": correct,
                          "total": n, "ci95": (lo, hi)}
    for field in SET_FIELDS:
        metrics[field] = multilabel(pred_rows, gold_rows, field)
    metrics["severity_ordinal"] = ordinal_metrics(pred_rows, gold_rows, "severity")
    metrics["claim_status_prf"] = per_class_prf(pred_rows, gold_rows, "claim_status",
                                                schema.CLAIM_STATUS)
    metrics["claim_status_confusion"] = confusion(pred_rows, gold_rows, "claim_status",
                                                  schema.CLAIM_STATUS)
    headline = [metrics[f]["accuracy"] for f in CATEGORICAL_FIELDS]
    headline += [metrics[f]["exact_set_accuracy"] for f in SET_FIELDS]
    metrics["_overall_mean"] = sum(headline) / len(headline)
    lo, hi = wilson_ci(round(metrics["_overall_mean"] * n), n)
    metrics["_overall_ci95"] = (lo, hi)
    return metrics


def error_rows(pred_rows, gold_rows, claim_rows) -> List[str]:
    """Compact per-row error listing for error analysis."""
    lines = []
    for i, (p, g) in enumerate(zip(pred_rows, gold_rows)):
        diffs = []
        for f in ["claim_status", "issue_type", "object_part", "severity",
                  "evidence_standard_met", "valid_image"]:
            if (p.get(f, "") or "").strip() != (g.get(f, "") or "").strip():
                diffs.append(f"{f}: {p.get(f)}≠{g.get(f)}")
        for f in SET_FIELDS:
            if _to_set(p.get(f, "")) != _to_set(g.get(f, "")):
                diffs.append(f"{f}: {p.get(f)}≠{g.get(f)}")
        if diffs:
            obj = claim_rows[i]["claim_object"] if claim_rows else ""
            uid = claim_rows[i]["user_id"] if claim_rows else f"row{i}"
            lines.append(f"- row {i} ({uid}, {obj}): " + "; ".join(diffs))
    return lines


# ---------------------------------------------------------------------------
# pretty printing
# ---------------------------------------------------------------------------
def format_metrics(metrics: Dict[str, dict]) -> str:
    lines = ["Field                       Accuracy        95% CI (Wilson)"]
    lines.append("-" * 60)
    for field in CATEGORICAL_FIELDS:
        m = metrics[field]
        lo, hi = m["ci95"]
        lines.append(f"{field:<26} {m['accuracy']*100:5.1f}% ({m['correct']}/{m['total']})"
                     f"   [{lo*100:.0f}-{hi*100:.0f}%]")
    for field in SET_FIELDS:
        m = metrics[field]
        lines.append(f"{field:<26} exact {m['exact_set_accuracy']*100:5.1f}%  "
                     f"micro-F1 {m['micro_f1']:.2f}  Hamming {m['hamming_per_claim']:.2f}/claim")
    lines.append("-" * 60)
    so = metrics["severity_ordinal"]
    lines.append(f"severity (ordinal): exact {so['exact']*100:.0f}%  "
                 f"within-1 {so['adjacent_within_1']*100:.0f}%  "
                 f"MAE {so['mae']:.2f}  QWK {so['qwk']:.2f}")
    cs = metrics["claim_status_prf"]
    lines.append(f"claim_status macro-F1 {cs['macro_f1']:.2f}  "
                 + "  ".join(f"{k}:F1={v['f1']:.2f}(n={v['support']})"
                             for k, v in cs["per_class"].items()))
    lines.append("-" * 60)
    lo, hi = metrics["_overall_ci95"]
    lines.append(f"{'OVERALL (mean of 8 fields)':<26} {metrics['_overall_mean']*100:5.1f}%"
                 f"   [{lo*100:.0f}-{hi*100:.0f}%]")
    return "\n".join(lines)


def format_confusion(metrics: Dict[str, dict]) -> str:
    labels = schema.CLAIM_STATUS
    m = metrics["claim_status_confusion"]
    short = {"supported": "supp", "contradicted": "contra", "not_enough_information": "NEI"}
    head = "gold\\pred   " + " ".join(f"{short[l]:>6}" for l in labels)
    rows = [head]
    for i, l in enumerate(labels):
        rows.append(f"{short[l]:<10} " + " ".join(f"{m[i][j]:>6}" for j in range(len(labels))))
    return "\n".join(rows)

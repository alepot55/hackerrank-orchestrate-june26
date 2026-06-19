"""Self-consistency ensembling.

The dominant source of "error wobble" on this task is the model's run-to-run
variance, not the prompt — repeated calls on the same claim disagree on the
harder fields (issue_type, severity). Self-consistency (Wang et al., 2022)
addresses exactly this: sample the model N times and take the majority vote per
field. Because the official scoring is field-by-field, we vote each field
INDEPENDENTLY (which maximises expected per-field accuracy); the two free-text
justifications are taken from a representative sample that agrees with the voted
claim_status.

This is a recognised accuracy/robustness technique, not extra machinery — it is
a thin voting layer over N model outputs, enabled only when ORCHESTRATE_SAMPLES
> 1.
"""
from __future__ import annotations

from collections import Counter
from typing import List

# Fields voted by simple majority (mode).
_CATEGORICAL = [
    "evidence_standard_met", "valid_image", "issue_type", "object_part",
    "claim_status", "severity",
]
# Multi-value fields voted element-wise.
_SET_FIELDS = ["visual_risk_flags", "supporting_image_ids"]
# Free-text fields taken from the representative sample.
_TEXT_FIELDS = ["reasoning", "evidence_standard_met_reason", "claim_status_justification"]

_CONF_RANK = {"high": 3, "medium": 2, "low": 1}


def _ordered(outputs: List[dict]) -> List[dict]:
    """Higher-confidence samples first, so majority ties break toward confidence."""
    return sorted(outputs, key=lambda o: _CONF_RANK.get(o.get("confidence"), 0), reverse=True)


def _majority(values):
    """Most common value; ties resolved by first occurrence in the (confidence-
    ordered) list."""
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def vote(outputs: List[dict]) -> dict:
    """Combine N model outputs into one via per-field majority vote."""
    outputs = [o for o in outputs if o]
    if not outputs:
        return {}
    if len(outputs) == 1:
        return outputs[0]
    ordered = _ordered(outputs)
    n = len(ordered)
    voted: dict = {}

    for field in _CATEGORICAL:
        vals = [o.get(field) for o in ordered if o.get(field) is not None]
        if vals:
            voted[field] = _majority(vals)

    # Element-wise majority for the multi-value fields: keep an item if it appears
    # in at least half of the samples.
    threshold = (n + 1) // 2
    for field in _SET_FIELDS:
        counts = Counter()
        for o in ordered:
            for item in (o.get(field) or []):
                counts[item] += 1
        voted[field] = [item for item, c in counts.items() if c >= threshold]

    # Representative sample for the free-text fields: the highest-confidence sample
    # whose claim_status matches the voted claim_status (falls back to the first).
    rep = next((o for o in ordered if o.get("claim_status") == voted.get("claim_status")),
               ordered[0])
    for field in _TEXT_FIELDS:
        voted[field] = rep.get(field)
    voted["confidence"] = rep.get("confidence")
    return voted

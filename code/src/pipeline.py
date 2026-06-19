"""Pipeline orchestration: read claims, run the model, merge the deterministic
history layer, validate against the allowed vocabularies, and write output.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from . import config, schema
from .history import history_risk_flags, load_user_history
from .runner import ModelRunner


def read_claims(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _b2s(value) -> str:
    return "true" if bool(value) else "false"


def _clamp(value, allowed, default):
    return value if value in allowed else default


def _order_visual_flags(flags) -> List[str]:
    present = {f for f in (flags or []) if f in schema.VISUAL_RISK_FLAGS}
    return [f for f in schema.VISUAL_RISK_FLAGS if f in present]


def finalize_row(claim_row: dict, model_out: dict, history_row: dict | None) -> dict:
    """Combine model output + deterministic rules into one validated output row."""
    claim_object = claim_row["claim_object"].strip()
    valid_parts = schema.OBJECT_PARTS.get(claim_object, ["unknown"])
    model_out = model_out or {}

    claim_status = _clamp(model_out.get("claim_status"), schema.CLAIM_STATUS,
                          "not_enough_information")
    issue_type = _clamp(model_out.get("issue_type"), schema.ISSUE_TYPE, "unknown")
    object_part = _clamp(model_out.get("object_part"), valid_parts, "unknown")
    severity = _clamp(model_out.get("severity"), schema.SEVERITY, "unknown")

    # Risk flags: visual (from model) + history (deterministic), ordered + deduped.
    risk = _order_visual_flags(model_out.get("visual_risk_flags"))
    risk += history_risk_flags(history_row)
    risk_flags = ";".join(risk) if risk else "none"

    # Supporting image IDs: keep only IDs that belong to this claim.
    claim_image_ids = {
        Path(p.strip()).stem for p in claim_row["image_paths"].split(";") if p.strip()
    }
    supporting = [i for i in (model_out.get("supporting_image_ids") or [])
                  if i in claim_image_ids]
    supporting_image_ids = ";".join(supporting) if supporting else "none"

    evidence_met = model_out.get("evidence_standard_met")
    valid_image = model_out.get("valid_image")
    # Safe fallbacks when the model returned nothing (e.g. a batch error).
    if not model_out:
        evidence_met, valid_image = False, False

    return {
        "user_id": claim_row["user_id"],
        "image_paths": claim_row["image_paths"],
        "user_claim": claim_row["user_claim"],
        "claim_object": claim_object,
        "evidence_standard_met": _b2s(evidence_met),
        "evidence_standard_met_reason": (
            model_out.get("evidence_standard_met_reason")
            or "Evidence assessed from the submitted images."
        ),
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": (
            model_out.get("claim_status_justification")
            or "Decision grounded in the submitted images."
        ),
        "supporting_image_ids": supporting_image_ids,
        "valid_image": _b2s(valid_image),
        "severity": severity,
    }


def run_pipeline(claims_path: Path, output_path: Path, use_batch: bool) -> dict:
    claims = read_claims(claims_path)
    history = load_user_history(config.USER_HISTORY_CSV)

    runner = ModelRunner(use_batch=use_batch)
    model_outputs: Dict[int, dict] = runner.run(claims)

    rows = [
        finalize_row(claim, model_outputs.get(idx, {}), history.get(claim["user_id"]))
        for idx, claim in enumerate(claims)
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    return runner.usage.as_dict(batch=use_batch)

"""Output schema, allowed value vocabularies, and the model's tool definition.

The output contract (column order and allowed values) is defined by
problem_statement.md. We expose it here as a single source of truth used by the
pipeline, the validation/clamping logic, and the evaluation harness.
"""
from __future__ import annotations

# Output CSV columns, in the exact required order.
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# The columns the *evaluation* compares (everything the system predicts).
PREDICTED_COLUMNS = [
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# ---------------------------------------------------------------------------
# Allowed value vocabularies
# ---------------------------------------------------------------------------
CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPE = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
]

SEVERITY = ["none", "low", "medium", "high", "unknown"]

# Object-specific part vocabularies (+ unknown).
OBJECT_PARTS = {
    "car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    ],
    "laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    ],
    "package": [
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    ],
}
# Union of every part value — used as a static enum so the tool definition stays
# byte-identical across all claims (preserving the prompt cache). We clamp to the
# object-specific list during post-processing.
ALL_OBJECT_PARTS = sorted({p for parts in OBJECT_PARTS.values() for p in parts})

# Risk flags the *model* may assign from the images (visual evidence only).
VISUAL_RISK_FLAGS = [
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
]
# Risk flags assigned deterministically from user history (never by the model).
HISTORY_RISK_FLAGS = ["user_history_risk", "manual_review_required"]

ALL_RISK_FLAGS = ["none"] + VISUAL_RISK_FLAGS + HISTORY_RISK_FLAGS


# ---------------------------------------------------------------------------
# Tool definition (structured output via forced tool use)
# ---------------------------------------------------------------------------
# Kept STATIC across every claim so it stays in the cached prompt prefix.
# `object_part` uses the union enum; we clamp per-object afterwards.
REVIEW_TOOL_NAME = "submit_review"

REVIEW_TOOL = {
    "name": REVIEW_TOOL_NAME,
    "description": (
        "Record the structured evidence review for one damage claim, grounded in "
        "the submitted images."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Brief visual analysis: what each image shows, whether the "
                    "claimed object/part is visible, and how it matches the claim. "
                    "2-4 sentences."
                ),
            },
            "evidence_standard_met": {
                "type": "boolean",
                "description": (
                    "True if the image set is sufficient to evaluate the claim "
                    "(claimed object and relevant part visible clearly enough)."
                ),
            },
            "evidence_standard_met_reason": {
                "type": "string",
                "description": "Short reason for the evidence_standard_met decision.",
            },
            "visual_risk_flags": {
                "type": "array",
                "items": {"type": "string", "enum": VISUAL_RISK_FLAGS},
                "description": (
                    "Image-derived risk flags only. Empty if none. Do NOT include "
                    "user-history flags."
                ),
            },
            "issue_type": {
                "type": "string",
                "enum": ISSUE_TYPE,
                "description": (
                    "Visible issue type. Use 'none' when the part is visible and "
                    "undamaged; 'unknown' when it cannot be determined."
                ),
            },
            "object_part": {
                "type": "string",
                "enum": ALL_OBJECT_PARTS,
                "description": "Relevant part of the claimed object (or 'unknown').",
            },
            "claim_status": {
                "type": "string",
                "enum": CLAIM_STATUS,
                "description": (
                    "supported (images confirm the claim), contradicted (images show "
                    "something different / no damage / wrong object), or "
                    "not_enough_information (claimed part not verifiable)."
                ),
            },
            "claim_status_justification": {
                "type": "string",
                "description": (
                    "Concise image-grounded explanation; mention relevant image IDs "
                    "when helpful."
                ),
            },
            "supporting_image_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Image IDs (e.g. 'img_1') that support the decision. Empty if no "
                    "image is sufficient."
                ),
            },
            "valid_image": {
                "type": "boolean",
                "description": (
                    "True if the image set is usable for automated review. False for "
                    "non-original/manipulated images or images too cropped/obstructed "
                    "to review."
                ),
            },
            "severity": {
                "type": "string",
                "enum": SEVERITY,
                "description": (
                    "Damage severity. 'none' when undamaged, 'unknown' when "
                    "not_enough_information, otherwise low/medium/high."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": (
                    "Your confidence in this overall decision given image quality "
                    "and clarity. Use 'low' when the images are ambiguous, the part "
                    "is hard to see, or the call is borderline."
                ),
            },
        },
        "required": [
            "reasoning", "evidence_standard_met", "evidence_standard_met_reason",
            "visual_risk_flags", "issue_type", "object_part", "claim_status",
            "claim_status_justification", "supporting_image_ids", "valid_image",
            "severity", "confidence",
        ],
    },
}

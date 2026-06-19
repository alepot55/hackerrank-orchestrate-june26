"""Deterministic user-history risk layer.

The user history risk flags (`user_history_risk`, `manual_review_required`) are a
pure lookup into user_history.csv — the images never decide them. Deriving them
in code (instead of asking the model) is cheaper, deterministic, and exactly
matches the labelled sample behaviour:

  * pass through whatever flags appear in the row's `history_flags`;
  * if `user_history_risk` is present, `manual_review_required` is also implied.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def load_user_history(path: Path) -> Dict[str, dict]:
    """Return {user_id: row_dict} from user_history.csv."""
    history: Dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            history[row["user_id"]] = row
    return history


def history_risk_flags(history_row: dict | None) -> List[str]:
    """Derive the ordered history risk flags for one claim's user."""
    if not history_row:
        return []
    raw = (history_row.get("history_flags") or "").strip()
    if not raw or raw == "none":
        return []
    flags = [f.strip() for f in raw.split(";") if f.strip() and f.strip() != "none"]
    flag_set = set(flags)
    # user_history_risk implies a manual review.
    if "user_history_risk" in flag_set:
        flag_set.add("manual_review_required")
    # Stable, deterministic order.
    order = ["user_history_risk", "manual_review_required"]
    return [f for f in order if f in flag_set]

"""Minimal .env loader (no external dependency).

Loads KEY=VALUE pairs from the repo-root .env into the environment if not already
set. The API key is therefore read from the environment only.
"""
from __future__ import annotations

import os

from . import config


def load_env() -> None:
    env_path = config.REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def require_api_key() -> None:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Copy code/.env.example to .env at the "
            "repository root and add your key, or export it in your shell."
        )

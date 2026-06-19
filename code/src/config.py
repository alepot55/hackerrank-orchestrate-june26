"""Central configuration: paths, model, cost controls, pricing constants.

All knobs are overridable via environment variables so the evaluator can run
the system without editing code. Secrets are read from the environment only.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# repo_root/code/src/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR  # image_paths in the CSVs are relative to dataset/

SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
TEST_CLAIMS_CSV = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQ_CSV = DATASET_DIR / "evidence_requirements.csv"

# Default output location for the test predictions.
OUTPUT_CSV = REPO_ROOT / "output.csv"

# On-disk cache of model responses, keyed by request hash. Re-runs are free.
CACHE_DIR = REPO_ROOT / "code" / ".cache"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL = os.environ.get("ORCHESTRATE_MODEL", "claude-opus-4-8")
MAX_TOKENS = int(os.environ.get("ORCHESTRATE_MAX_TOKENS", "1200"))

# ---------------------------------------------------------------------------
# Cost controls
# ---------------------------------------------------------------------------
# Downscale images so the long edge is at most this many pixels before sending.
# Smaller -> fewer image tokens -> cheaper. 1456 keeps damage detail readable
# while staying well under the high-resolution token blow-up.
IMAGE_MAX_EDGE = int(os.environ.get("ORCHESTRATE_IMAGE_MAX_EDGE", "1456"))
IMAGE_JPEG_QUALITY = int(os.environ.get("ORCHESTRATE_IMAGE_QUALITY", "80"))

# Use the asynchronous Batch API for the (latency-insensitive) test run: 50% off.
USE_BATCH = os.environ.get("ORCHESTRATE_USE_BATCH", "1") not in ("0", "false", "False")

# Prompt-cache TTL. "1h" is right for batch runs that span more than 5 minutes.
CACHE_TTL = os.environ.get("ORCHESTRATE_CACHE_TTL", "1h")


# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens) — used only for the operational cost report.
# Source: Claude Opus 4.8 standard pricing.
# ---------------------------------------------------------------------------
PRICE_INPUT_PER_MTOK = 5.0
PRICE_OUTPUT_PER_MTOK = 25.0
PRICE_CACHE_WRITE_1H_MULT = 2.0   # 1-hour cache write multiplier on input price
PRICE_CACHE_WRITE_5M_MULT = 1.25  # 5-minute cache write multiplier
PRICE_CACHE_READ_MULT = 0.1       # cache read multiplier on input price
BATCH_DISCOUNT = 0.5              # Batch API: 50% off input and output

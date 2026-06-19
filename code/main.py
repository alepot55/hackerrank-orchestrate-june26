#!/usr/bin/env python3
"""Entry point: run the evidence-review system on dataset/claims.csv.

Produces output.csv at the repository root with one structured prediction per
claim. Uses the Batch API by default (50% cheaper) since the test run is not
latency-sensitive; set ORCHESTRATE_USE_BATCH=0 to run synchronously.

Usage:
    python code/main.py [--sync] [--output PATH]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `src` importable

from src import config, envload, pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence review on claims.csv")
    parser.add_argument("--sync", action="store_true",
                        help="Run synchronously instead of the Batch API.")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_CSV,
                        help="Output CSV path (default: repo-root output.csv).")
    args = parser.parse_args()

    envload.require_api_key()
    use_batch = config.USE_BATCH and not args.sync

    print(f"Model: {config.MODEL} | mode: {'batch' if use_batch else 'sync'} | "
          f"image max edge: {config.IMAGE_MAX_EDGE}px")
    print(f"Reading claims from {config.TEST_CLAIMS_CSV}")

    start = time.time()
    usage = pipeline.run_pipeline(config.TEST_CLAIMS_CSV, args.output, use_batch)
    elapsed = time.time() - start

    print(f"\nWrote predictions to {args.output}")
    print(f"Runtime: {elapsed:.1f}s")
    print("Usage / cost:")
    for k, v in usage.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

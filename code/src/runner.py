"""Model runner: builds requests, calls Claude, caches, and tracks cost.

Cost-saving features baked in:
  * On-disk response cache keyed by the full request hash — re-running the
    pipeline (or re-evaluating) costs nothing for already-seen claims.
  * Prompt caching (cache_control on the static system+tools prefix).
  * Optional Batch API path (50% cheaper) for the latency-insensitive test run.
  * Structured output via forced tool use — guarantees a schema-valid result.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import anthropic

from . import config, prompts, schema
from .images import encode_image


class Usage:
    """Accumulates token usage and estimates cost for the operational report."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_write_tokens = 0
        self.cache_read_tokens = 0
        self.api_calls = 0          # billed model calls actually made
        self.cache_hits = 0         # served from the local disk cache (free)
        self.images = 0

    def add(self, u) -> None:
        if u is None:
            return
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0

    def estimate_cost_usd(self, batch: bool) -> float:
        disc = config.BATCH_DISCOUNT if batch else 1.0
        write_mult = (config.PRICE_CACHE_WRITE_1H_MULT if config.CACHE_TTL == "1h"
                      else config.PRICE_CACHE_WRITE_5M_MULT)
        inp = config.PRICE_INPUT_PER_MTOK / 1e6
        out = config.PRICE_OUTPUT_PER_MTOK / 1e6
        cost = (
            self.input_tokens * inp
            + self.cache_write_tokens * inp * write_mult
            + self.cache_read_tokens * inp * config.PRICE_CACHE_READ_MULT
            + self.output_tokens * out
        )
        return cost * disc

    def as_dict(self, batch: bool) -> dict:
        return {
            "billed_api_calls": self.api_calls,
            "local_cache_hits": self.cache_hits,
            "images_processed": self.images,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "estimated_cost_usd": round(self.estimate_cost_usd(batch), 4),
        }


class ModelRunner:
    def __init__(self, use_batch: bool = False) -> None:
        # Reads ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.Anthropic(max_retries=4)
        self.use_batch = use_batch
        self.usage = Usage()
        self._system = prompts.build_system_prompt(
            prompts.load_evidence_requirements(config.EVIDENCE_REQ_CSV)
        )
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -- request building ---------------------------------------------------
    def build_params(self, claim_row: dict) -> Tuple[dict, List[str]]:
        image_ids = prompts.image_ids_for_claim(claim_row["image_paths"])
        from .images import split_image_paths
        blocks, present_ids = [], []
        for rel, img_id in zip(split_image_paths(claim_row["image_paths"]), image_ids):
            b = encode_image(rel)
            if b is not None:
                blocks.append(b)
                present_ids.append(img_id)
        user_content = prompts.build_user_content(claim_row, blocks, present_ids)
        # Prompt caching only helps the SYNC path, where calls are sequential and
        # later claims read the cached prefix. In BATCH mode the 44 requests run
        # concurrently, so none can read another's cache — they would only pay the
        # cache-write premium. So we cache for sync and skip it for batch.
        system_block = {"type": "text", "text": self._system}
        if not self.use_batch:
            system_block["cache_control"] = {"type": "ephemeral"}  # 5-min TTL
        params = {
            "model": config.MODEL,
            "max_tokens": config.MAX_TOKENS,
            "system": [system_block],
            "tools": [schema.REVIEW_TOOL],
            "tool_choice": {"type": "tool", "name": schema.REVIEW_TOOL_NAME},
            "messages": [{"role": "user", "content": user_content}],
        }
        return params, present_ids

    # -- disk cache ---------------------------------------------------------
    def _cache_path(self, params: dict) -> Path:
        key = json.dumps(params, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()[:32]
        return config.CACHE_DIR / f"{digest}.json"

    @staticmethod
    def _extract_tool_input(content) -> dict | None:
        for block in content:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if btype == "tool_use":
                name = getattr(block, "name", None) or block.get("name")
                if name == schema.REVIEW_TOOL_NAME:
                    return getattr(block, "input", None) or block.get("input")
        return None

    # -- execution ----------------------------------------------------------
    def run(self, claim_rows: List[dict]) -> Dict[int, dict]:
        """Return {row_index: model_output_dict} for every claim."""
        prepared = []  # (idx, params, image_ids, cache_path)
        for idx, row in enumerate(claim_rows):
            params, image_ids = self.build_params(row)
            self.usage.images += len(image_ids)
            prepared.append((idx, params, image_ids, self._cache_path(params)))

        results: Dict[int, dict] = {}
        pending = []  # (idx, params, cache_path)
        for idx, params, image_ids, cpath in prepared:
            if cpath.exists():
                cached = json.loads(cpath.read_text())
                results[idx] = cached["output"]
                self.usage.cache_hits += 1
            else:
                pending.append((idx, params, cpath))

        if pending:
            if self.use_batch:
                self._run_batch(pending, results)
            else:
                self._run_sync(pending, results)
        return results

    def _store(self, cpath: Path, output: dict) -> None:
        cpath.write_text(json.dumps({"output": output}, ensure_ascii=False))

    def _run_sync(self, pending, results) -> None:
        for idx, params, cpath in pending:
            resp = self.client.messages.create(**params)
            self.usage.add(resp.usage)
            self.usage.api_calls += 1
            out = self._extract_tool_input(resp.content) or {}
            results[idx] = out
            self._store(cpath, out)

    def _run_batch(self, pending, results) -> None:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        index_by_cid = {f"claim-{idx}": (idx, cpath) for idx, _, cpath in pending}
        requests = [
            Request(custom_id=f"claim-{idx}", params=MessageCreateParamsNonStreaming(**params))
            for idx, params, _ in pending
        ]
        batch = self.client.messages.batches.create(requests=requests)
        print(f"  Batch {batch.id} submitted ({len(requests)} requests). Polling...")
        while True:
            b = self.client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(15)
        for r in self.client.messages.batches.results(batch.id):
            idx, cpath = index_by_cid[r.custom_id]
            if r.result.type == "succeeded":
                msg = r.result.message
                self.usage.add(msg.usage)
                self.usage.api_calls += 1
                out = self._extract_tool_input(msg.content) or {}
            else:
                out = {}  # errored/expired -> empty; post-processing fills safe defaults
            results[idx] = out
            self._store(cpath, out)

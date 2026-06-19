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

from . import config, ensemble, prompts, schema
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
    def build_params(self, claim_row: dict, sample_idx: int = 0) -> Tuple[dict, List[str]]:
        image_ids = prompts.image_ids_for_claim(claim_row["image_paths"])
        from .images import split_image_paths
        blocks, present_ids = [], []
        for rel, img_id in zip(split_image_paths(claim_row["image_paths"]), image_ids):
            b = encode_image(rel)
            if b is not None:
                blocks.append(b)
                present_ids.append(img_id)
        user_content = prompts.build_user_content(claim_row, blocks, present_ids)
        # For self-consistency we need N distinct samples (and distinct cache keys).
        # A neutral marker on the header makes each sample independent without
        # biasing the visual judgement.
        if sample_idx > 0 and user_content and user_content[0].get("type") == "text":
            first = dict(user_content[0])
            first["text"] = first["text"] + f"\n\n(Independent assessment #{sample_idx + 1}.)"
            user_content = [first] + user_content[1:]
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
        # Key on the *semantic* request only — strip cache_control (a billing knob
        # that does not change the model's answer) so toggling caching/TTL does
        # not invalidate the on-disk result cache.
        norm = {
            "model": params["model"],
            "max_tokens": params["max_tokens"],
            "tools": params["tools"],
            "tool_choice": params["tool_choice"],
            "messages": params["messages"],
            "system": [b.get("text") for b in params["system"]],
        }
        key = json.dumps(norm, sort_keys=True).encode("utf-8")
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
        """Return {row_index: model_output_dict} for every claim.

        With config.SAMPLES > 1 each claim is sampled N times and the per-field
        majority vote is returned (self-consistency).
        """
        samples = max(1, config.SAMPLES)
        # Build one request per (claim, sample). Key requests by (idx, sample).
        prepared = []  # (key, params, cache_path)
        for idx, row in enumerate(claim_rows):
            for s in range(samples):
                params, image_ids = self.build_params(row, s)
                if s == 0:
                    self.usage.images += len(image_ids)
                prepared.append(((idx, s), params, self._cache_path(params)))

        raw: Dict[tuple, dict] = {}
        pending = []  # (key, params, cache_path)
        for key, params, cpath in prepared:
            if cpath.exists():
                raw[key] = json.loads(cpath.read_text())["output"]
                self.usage.cache_hits += 1
            else:
                pending.append((key, params, cpath))

        if pending:
            if self.use_batch:
                self._run_batch(pending, raw)
            else:
                self._run_sync(pending, raw)

        # Collapse the N samples per claim into one output via majority vote.
        results: Dict[int, dict] = {}
        for idx in range(len(claim_rows)):
            outs = [raw.get((idx, s), {}) for s in range(samples)]
            results[idx] = ensemble.vote(outs) if samples > 1 else outs[0]
        return results

    def _store(self, cpath: Path, output: dict) -> None:
        cpath.write_text(json.dumps({"output": output}, ensure_ascii=False))

    def _run_sync(self, pending, raw) -> None:
        for key, params, cpath in pending:
            resp = self.client.messages.create(**params)
            self.usage.add(resp.usage)
            self.usage.api_calls += 1
            out = self._extract_tool_input(resp.content) or {}
            if config.ESCALATE and out.get("confidence") == "low":
                out = self._grounded_repass(params, out)
            raw[key] = out
            if out:  # never cache an empty/failed result — let it retry next run
                self._store(cpath, out)

    def _grounded_repass(self, params: dict, first: dict) -> dict:
        """Second, grounded re-examination for a low-confidence claim. We append a
        focused-inspection nudge (not a naive 'are you sure?') and re-decide;
        intrinsic self-doubt is avoided per the self-correction literature."""
        nudge = {
            "role": "user",
            "content": (
                "Re-examine ONLY the claimed object part. Look closely at the "
                "relevant region of each image and decide strictly from what is "
                "actually visible there. If the claimed damage is not clearly "
                "visible, do not assume it exists. Re-issue your structured review."
            ),
        }
        repass = dict(params)
        repass["messages"] = params["messages"] + [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "prior", "name": schema.REVIEW_TOOL_NAME,
                 "input": first}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "prior", "content": "noted"}]},
            nudge,
        ]
        resp = self.client.messages.create(**repass)
        self.usage.add(resp.usage)
        self.usage.api_calls += 1
        return self._extract_tool_input(resp.content) or first

    def _run_batch(self, pending, raw) -> None:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        def cid(key):  # (idx, sample) -> valid custom_id ^[a-zA-Z0-9_-]{1,64}$
            return f"c{key[0]}-s{key[1]}"

        by_cid = {cid(key): (key, cpath) for key, _, cpath in pending}
        requests = [
            Request(custom_id=cid(key), params=MessageCreateParamsNonStreaming(**params))
            for key, params, _ in pending
        ]
        batch = self.client.messages.batches.create(requests=requests)
        print(f"  Batch {batch.id} submitted ({len(requests)} requests). Polling...")
        while True:
            b = self.client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(15)
        for r in self.client.messages.batches.results(batch.id):
            key, cpath = by_cid[r.custom_id]
            if r.result.type == "succeeded":
                msg = r.result.message
                self.usage.add(msg.usage)
                self.usage.api_calls += 1
                out = self._extract_tool_input(msg.content) or {}
                raw[key] = out
                if out:  # only cache successful, non-empty results
                    self._store(cpath, out)
            else:
                # errored/expired -> empty; post-processing fills safe defaults.
                # Do NOT cache it, so a re-run retries the failed request.
                raw[key] = {}

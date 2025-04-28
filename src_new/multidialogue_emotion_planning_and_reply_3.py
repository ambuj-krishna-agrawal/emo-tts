#!/usr/bin/env python3
"""
End-to-end pipeline (v3.0 – 2025-04-22)
======================================
This revision refactors the pipeline to use the modular emotion_detection_and_response
module for better code organization and reusability. The core functionality remains
the same, but now it can be more easily reused in other applications like the Gradio demo.

Key changes vs v2.1
-------------------
1. **Modular architecture** – Core functions moved to emotion_detection_and_response.py
2. **Import-based approach** – Using detect_emotion and generate_*_response functions
3. **Same behavior** – Maintains all the functionality of v2.1 with cleaner structure
4. **Better error handling** – More robust handling of errors in the model calls
5. **Separated mapping from downsampling** – Added CLI flag to disable downsampling
6. **Extensive logging** – Tracks loading, filtering, downsampling, and missing entries
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm

# Import our emotion detection and response module
from src_new.emotion_detection_and_response import (
    setup_logging,
    detect_emotion,
    generate_emotional_response,
    generate_neutral_response,
    map_emotion,
    standardize_reference_emotion,
    create_inference_client
)

# ------------------- Constants --------------------
DEFAULT_MODEL = "llama_3_70b_q4"
PAIRS_METADATA = Path(
    "/data/group_data/starlight/gpa/tts/multidialog_sds_pairs/pairs_metadata.json"
)
OUTPUT_DIR = Path(
    f"/data/group_data/starlight/gpa/tts/multidialog_emotion_planning/{DEFAULT_MODEL}"
)
MAX_WORKERS = 4
BATCH_SIZE = 8
RANDOM_SEED = 42
LENGTH_RATIO_DEFAULT = 0.9
TEMPERATURE_GEN_DEFAULT = 0.7

# ------------------- Data Classes --------------------

@dataclass
class EntryResult:
    pair_id: str
    history: str
    speaker_name: str
    target_emotion: str
    emotion_steered_reply: str
    baseline_reply: str
    reference_emotion: str
    standardized_reference_emotion: str
    reference_response: str
    timestamp: str

# ------------------- Core pipeline --------------------

def generate_responses_batch(
    router,
    alias: str,
    entries: List[Dict[str, Any]],
    logger: logging.Logger,
    gen_temperature: float,
    length_ratio: float
):
    logger.info(f"Processing batch of {len(entries)} entries")
    
    # Detect emotions
    emotions: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pid = {
            executor.submit(detect_emotion, e["text_A"], alias): e["pair_id"]
            for e in entries
        }
        with tqdm(total=len(entries), desc="Classifying") as pbar:
            for f in as_completed(future_to_pid):
                pid = future_to_pid[f]
                try:
                    emotions[pid] = f.result()
                except Exception as e:
                    logger.error(f"Error detecting emotion for {pid}: {e}")
                    emotions[pid] = "Neutral"
                pbar.update(1)

    # Generate steered replies
    steered_results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pid = {
            executor.submit(
                generate_emotional_response,
                e["text_A"],
                emotions.get(e["pair_id"], "Neutral"),
                length_ratio,
                alias,
                gen_temperature
            ): e["pair_id"]
            for e in entries
        }
        with tqdm(total=len(entries), desc="Steered") as pbar:
            for f in as_completed(future_to_pid):
                pid = future_to_pid[f]
                try:
                    steered_results[pid] = f.result()
                except Exception as e:
                    logger.error(f"Error generating steered response for {pid}: {e}")
                    steered_results[pid] = f"Error: {e}"
                pbar.update(1)

    # Generate baseline replies
    baseline_results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pid = {
            executor.submit(
                generate_neutral_response,
                e["text_A"],
                length_ratio,
                alias,
                gen_temperature
            ): e["pair_id"]
            for e in entries
        }
        with tqdm(total=len(entries), desc="Baseline") as pbar:
            for f in as_completed(future_to_pid):
                pid = future_to_pid[f]
                try:
                    baseline_results[pid] = f.result()
                except Exception as e:
                    logger.error(f"Error generating baseline response for {pid}: {e}")
                    baseline_results[pid] = f"Error: {e}"
                pbar.update(1)

    # Compile results
    results: List[Dict[str, Any]] = []
    for e in entries:
        pid = e["pair_id"]
        ref_emo = map_emotion(e.get("emotion_B", ""))
        std_ref = standardize_reference_emotion(e.get("emotion_B", ""))
        results.append(vars(EntryResult(
            pair_id=pid,
            history=e["text_A"],
            speaker_name=e["speaker_B"],
            target_emotion=emotions.get(pid, "Neutral"),
            emotion_steered_reply=steered_results.get(pid, "(no response)"),
            baseline_reply=baseline_results.get(pid, "(no response)"),
            reference_emotion=ref_emo,
            standardized_reference_emotion=std_ref,
            reference_response=e["text_B"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )))
    return results

# ------------------- Dataset helpers --------------------

def load_pairs(fp: Path) -> List[Dict[str, Any]]:
    try:
        return json.loads(fp.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed loading {fp}: {exc}")

def map_emotions(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for p in pairs:
        p["reference_emotion"] = map_emotion(p.get("emotion_B", ""))
        p["standardized_reference_emotion"] = standardize_reference_emotion(p.get("emotion_B", ""))
    return pairs

def downsample_excited(
    pairs: List[Dict[str, Any]],
    seed: int = RANDOM_SEED,
    logger: logging.Logger = None
) -> List[Dict[str, Any]]:
    """
    If Excited is over‐represented, sample it down by pair_id to match the largest other class.
    Preserves original ordering of kept examples.
    """
    freqs = Counter(p["standardized_reference_emotion"] for p in pairs)
    excited_cnt = freqs.get("Excited", 0)
    max_other = max((cnt for emo, cnt in freqs.items() if emo != "Excited"), default=0)
    if excited_cnt <= max_other:
        if logger:
            logger.info("No downsampling needed (Excited=%d, max_other=%d)", excited_cnt, max_other)
        return pairs

    # Separate excited vs others
    excited = [p for p in pairs if p["standardized_reference_emotion"] == "Excited"]
    others = [p for p in pairs if p["standardized_reference_emotion"] != "Excited"]

    # Sample by pair_id
    random.seed(seed)
    excited_ids = [p["pair_id"] for p in excited]
    sampled_ids = set(random.sample(excited_ids, max_other))
    removed_ids = set(excited_ids) - sampled_ids

    if logger:
        logger.info(
            "Downsampling Excited: total_excited=%d → keep=%d, remove=%d",
            excited_cnt, len(sampled_ids), len(removed_ids)
        )
        logger.debug("Removed Excited IDs: %s", sorted(removed_ids))

    # Reassemble in original order
    new_pairs: List[Dict[str, Any]] = []
    for p in pairs:
        if p["standardized_reference_emotion"] != "Excited" or p["pair_id"] in sampled_ids:
            new_pairs.append(p)
    return new_pairs

def filter_top_n_by_length(pairs: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return sorted(pairs, key=lambda p: len(p.get("text_A", "")), reverse=True)[:n]

# ------------------- Main --------------------

def main():
    p = argparse.ArgumentParser(description="Emotion-aware reply generator (modular version)")
    p.add_argument("--input", "-i", default=str(PAIRS_METADATA), help="Input JSON path")
    p.add_argument("--output-dir", "-o", default=str(OUTPUT_DIR), help="Output directory")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model alias")
    p.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help="Threads per batch")
    p.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE, help="Batch size")
    p.add_argument("--filter-top", "-f", type=int, default=0, help="Top-N filter by length")
    p.add_argument("--no-downsample", action="store_true", help="Skip downsampling")
    p.add_argument("--gen-temperature", type=float, default=TEMPERATURE_GEN_DEFAULT, help="Generation temperature")
    p.add_argument("--length-ratio", type=float, default=LENGTH_RATIO_DEFAULT, help="Reply length ratio")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "emotion_reply.log")
    logger.info("Args: %s", vars(args))

    # Load and log initial set of pair IDs
    pairs = load_pairs(Path(args.input))
    initial_ids = {p["pair_id"] for p in pairs}
    logger.info("Loaded %d pairs (sample IDs %s…)", len(initial_ids), list(initial_ids)[:5])

    # Optional length-based filtering
    if args.filter_top > 0:
        logger.info("Filtering top-%d by length… before=%d", args.filter_top, len(pairs))
        kept = filter_top_n_by_length(pairs, args.filter_top)
        removed = {p["pair_id"] for p in pairs} - {p["pair_id"] for p in kept}
        logger.info(" → kept=%d, removed=%d (sample removed IDs %s)", len(kept), len(removed), sorted(removed)[:5])
        pairs = kept

    # Map reference emotions
    pairs = map_emotions(pairs)
    logger.debug(
        "After mapping emotions, sample of reference_emotion: %s",
        [(p["pair_id"], p["reference_emotion"]) for p in pairs[:5]]
    )

    # Optional downsampling of "Excited"
    if not args.no_downsample:
        pairs = downsample_excited(pairs, seed=RANDOM_SEED, logger=logger)
        logger.info("After downsampling: %d pairs remain", len(pairs))

    logger.info("Dataset after preprocessing: %d pairs", len(pairs))

    # Create inference client
    client = create_inference_client()
    if not client:
        logger.error("Failed to create inference client")
        sys.exit(1)

    # Batchwise response generation
    from math import ceil
    total_batches = ceil(len(pairs) / args.batch_size)
    results: List[Dict[str, Any]] = []
    for i in tqdm(range(0, len(pairs), args.batch_size), total=total_batches, desc="Overall", unit="batch"):
        batch = pairs[i : i + args.batch_size]
        logger.info("Batch %d/%d", i // args.batch_size + 1, total_batches)
        results.extend(
            generate_responses_batch(
                client, args.model, batch, logger, args.gen_temperature, args.length_ratio
            )
        )
        # Periodic partial dumps
        if (i // args.batch_size) % 5 == 0:
            tmp = out_dir / f"partial_{len(results)}.jsonl"
            tmp.write_text("\n".join(json.dumps(r) for r in results))

    # Final outputs
    (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results))
    stats = {
        "total": len(results),
        "emotion_distribution": Counter(r["target_emotion"] for r in results),
        "agreement": sum(1 for r in results if r["target_emotion"] == r["standardized_reference_emotion"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    stats["agreement_pct"] = 100 * stats["agreement"] / stats["total"] if stats["total"] else 0
    (out_dir / "stats.json").write_text(
        json.dumps({k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}, indent=2)
    )

    # Check for any missing IDs
    processed_ids = {r["pair_id"] for r in results}
    missing = initial_ids - processed_ids
    if missing:
        logger.warning(
            "Some pair_ids were never processed (%d): %s",
            len(missing), sorted(missing)[:10]
        )

    logger.info("Done – processed %d pairs", len(results))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

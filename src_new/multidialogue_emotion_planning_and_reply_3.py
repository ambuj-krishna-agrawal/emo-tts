#!/usr/bin/env python3
"""
End‑to‑end pipeline (v3.0 – 2025‑04‑22)
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

Run example
-----------
```bash
python multidialog_emotion_planning_and_reply_3.py \
    --length-ratio 0.9 \
    --input path/to/pairs.json \
    --output-dir results/run3
``` 
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
from typing import Any, Dict, List, Tuple

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
OUTPUT_DIR = Path(f"/data/group_data/starlight/gpa/tts/multidialog_emotion_planning/{DEFAULT_MODEL}")
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
    """
    Generate emotion detection and responses for a batch of entries
    
    Args:
        router: The inference client
        alias: The model alias to use
        entries: List of entries containing text_A and other fields
        logger: Logger instance
        gen_temperature: Temperature for generation
        length_ratio: Ratio of response length to input length
        
    Returns:
        List of result dictionaries
    """
    logger.info(f"Processing batch of {len(entries)} entries")
    
    # First, detect emotions for all entries
    emotions = {}
    logger.info(f"Detecting emotions for {len(entries)} pairs")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pid = {}
        for entry in entries:
            pid = entry["pair_id"]
            
            # Create a future for each emotion detection task
            future_to_pid[executor.submit(
                detect_emotion, 
                entry["text_A"],
                alias
            )] = pid
        
        # Process results as they complete
        with tqdm(total=len(entries), desc="Classifying") as pbar:
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    emotions[pid] = future.result()
                except Exception as e:
                    logger.error(f"Error detecting emotion for {pid}: {e}")
                    emotions[pid] = "Neutral"  # Default to Neutral on error
                pbar.update(1)
    
    # Generate emotional responses
    steered_results = {}
    logger.info(f"Generating {len(entries)} steered replies")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pid = {}
        for entry in entries:
            pid = entry["pair_id"]
            target_emotion = emotions.get(pid, "Neutral")
            
            # Create a future for each response generation task
            future_to_pid[executor.submit(
                generate_emotional_response,
                entry["text_A"],
                target_emotion,
                length_ratio,
                alias,
                gen_temperature
            )] = pid
        
        # Process results as they complete
        with tqdm(total=len(entries), desc="Steered") as pbar:
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    steered_results[pid] = future.result()
                except Exception as e:
                    logger.error(f"Error generating steered response for {pid}: {e}")
                    steered_results[pid] = f"Error: {e}"
                pbar.update(1)
    
    # Generate neutral responses
    baseline_results = {}
    logger.info(f"Generating {len(entries)} baseline replies")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pid = {}
        for entry in entries:
            pid = entry["pair_id"]
            
            # Create a future for each neutral response generation task
            future_to_pid[executor.submit(
                generate_neutral_response,
                entry["text_A"],
                length_ratio,
                alias,
                gen_temperature
            )] = pid
        
        # Process results as they complete
        with tqdm(total=len(entries), desc="Baseline") as pbar:
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    baseline_results[pid] = future.result()
                except Exception as e:
                    logger.error(f"Error generating baseline response for {pid}: {e}")
                    baseline_results[pid] = f"Error: {e}"
                pbar.update(1)
    
    # Compile results
    results = []
    for entry in entries:
        pid = entry["pair_id"]
        target = emotions.get(pid, "Neutral")
        ref_emo = map_emotion(entry.get("emotion_B", ""))
        std_ref = standardize_reference_emotion(entry.get("emotion_B", ""))
        
        results.append(vars(EntryResult(
            pair_id=pid,
            history=entry["text_A"],
            speaker_name=entry["speaker_B"],
            target_emotion=target,
            emotion_steered_reply=steered_results.get(pid, "(no response)"),
            baseline_reply=baseline_results.get(pid, "(no response)"),
            reference_emotion=ref_emo,
            standardized_reference_emotion=std_ref,
            reference_response=entry["text_B"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )))
    
    return results

# ------------------- Dataset helpers --------------------

def load_pairs(fp: Path):
    """Load pairs from a JSON file"""
    try:
        return json.loads(fp.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed loading {fp}: {exc}")

def remap_and_downsample(pairs):
    """Remap emotions and downsample to balance the dataset"""
    for p in pairs:
        p["reference_emotion"] = map_emotion(p.get("emotion_B", ""))
        p["standardized_reference_emotion"] = standardize_reference_emotion(p.get("emotion_B", ""))
    
    freqs = Counter(p["standardized_reference_emotion"] for p in pairs)
    excited_cnt = freqs.get("Excited", 0)
    max_other = max(cnt for emo, cnt in freqs.items() if emo != "Excited") if freqs else 0
    
    if excited_cnt > max_other:
        random.seed(RANDOM_SEED)
        excited = [p for p in pairs if p["standardized_reference_emotion"] == "Excited"]
        others = [p for p in pairs if p["standardized_reference_emotion"] != "Excited"]
        pairs = others + random.sample(excited, max_other)
        random.shuffle(pairs)
    
    return pairs

def filter_top_n_by_length(pairs, n: int):
    """Filter and return the top n pairs by text_A length"""
    return sorted(pairs, key=lambda p: len(p.get("text_A", "")), reverse=True)[:n]

# ------------------- Main --------------------

def main():
    p = argparse.ArgumentParser(description="Emotion‑aware reply generator (modular version)")
    p.add_argument("--input", "-i", default=str(PAIRS_METADATA))
    p.add_argument("--output-dir", "-o", default=str(OUTPUT_DIR))
    p.add_argument("--model", "-m", default=DEFAULT_MODEL)
    p.add_argument("--workers", "-w", type=int, default=MAX_WORKERS)
    p.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE)
    p.add_argument("--filter-top", "-f", type=int, default=0)
    p.add_argument("--gen-temperature", type=float, default=TEMPERATURE_GEN_DEFAULT)
    p.add_argument("--length-ratio", type=float, default=LENGTH_RATIO_DEFAULT, help="Reply length = ratio × len(text_A) in words")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "emotion_reply.log")
    logger.info("Args: %s", vars(args))

    pairs = load_pairs(Path(args.input))
    if args.filter_top > 0:
        pairs = filter_top_n_by_length(pairs, args.filter_top)
    pairs = remap_and_downsample(pairs)
    logger.info("Dataset after preprocessing: %d pairs", len(pairs))

    client = create_inference_client()
    if not client:
        logger.error("Failed to create inference client")
        sys.exit(1)
        
    results, batch_size = [], args.batch_size
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        logger.info("Batch %d/%d", i // batch_size + 1, (len(pairs)+batch_size-1)//batch_size)
        results.extend(generate_responses_batch(client, args.model, batch, logger, args.gen_temperature, args.length_ratio))
        
        # Save partial results every 5 batches
        if (i // batch_size) % 5 == 0:
            tmp = out_dir / f"partial_{len(results)}.jsonl"
            tmp.write_text("\n".join(json.dumps(r) for r in results))
    
    # Save final results
    (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results))

    # Calculate and save statistics
    stats = {
        "total": len(results),
        "emotion_distribution": Counter(r["target_emotion"] for r in results),
        "agreement": sum(1 for r in results if r["target_emotion"] == r["standardized_reference_emotion"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    stats["agreement_pct"] = 100 * stats["agreement"] / stats["total"] if stats["total"] else 0
    
    (out_dir / "stats.json").write_text(json.dumps({
        k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()
    }, indent=2))
    
    logger.info("Done – processed %d pairs", len(results))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
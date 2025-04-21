#!/usr/bin/env python3
"""
summarizer.py
───────────────────────────────────────────────────────────────────────────────
Process VERSA TTS evaluation results from auto_summary.jsonl, which is a JSONL file
containing merged per-utterance metrics and a final "overall" summary entry.

This script:
1. Reads the auto_summary.jsonl file (default: evaluation_results/auto_summary.jsonl)
2. Extracts the "overall" metrics
3. Writes the overall metrics to evaluation_results/metrics_summary.json
4. Prints the summary to the console
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

# Default paths
INPUT_PATH = Path("evaluation_results/auto_summary.jsonl")
OUTPUT_DIR = Path("evaluation_results")
SUMMARY_PATH = OUTPUT_DIR / "metrics_summary.json"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file where each line is a JSON object.

    Returns:
        List of dicts parsed from each non-empty line.
    """
    data: List[Dict[str, Any]] = []
    with path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON line: {line} ({e})")
    return data


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the JSONL data
    logging.info(f"Reading auto summary from {INPUT_PATH}")
    records = read_jsonl(INPUT_PATH)
    if not records:
        logging.error("No records found in input file.")
        return

    # Extract the overall entry
    overall = next((r for r in records if r.get("key") == "overall"), None)
    if not overall:
        logging.error("No 'overall' entry found in JSONL.")
        return

    # Remove the 'key' field for summary
    summary: Dict[str, Any] = {k: v for k, v in overall.items() if k != "key"}

    # Write summary JSON
    with SUMMARY_PATH.open('w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary metrics saved to {SUMMARY_PATH}")

    # Print summary
    print("\n===== VERSA TTS Evaluation Summary =====")
    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()

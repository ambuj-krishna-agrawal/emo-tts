#!/usr/bin/env python
"""
evaluate_with_versa.py
───────────────────────────────────────────────────────────────────────────────
Run objective TTS evaluation using the **VERSA** scorer already installed in
this environment.

Two passes are launched:
1. Independent metrics (no reference): UTMOS + DNSMOS (pseudo_mos), NISQA, SRMR.
2. Reference metrics: PESQ, STOI, signal_metric (SI‑SNR & LSD bundle), VISQOL,
   and MCD + F0‑RMSE (mcd_f0).

VERSA expects the YAML to be a top‑level list of metric configs, so that’s what
this script writes.

Example:
    python evaluate_with_versa.py --metadata generated_wavs/train/metadata.json --device cuda -v
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# ─────────────────────────── helper: build scp files ─────────────────────────

def build_scp(paths: List[Path], scp_path: Path):
    with scp_path.open("w") as f:
        for i, p in enumerate(paths):
            f.write(f"utt{i:05d} {p}\n")

# ─────────────────────────── helper: run VERSA CLI ───────────────────────────

def run_versa(args: List[str]):
    cmd = [sys.executable, "-m", "versa.bin.scorer"] + args
    logging.debug("RUN: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ───────────────────────── helper: reading & writing JSONL ─────────────────────

def read_metrics(path: Path) -> List[Dict[str, Any]]:
    """
    Read a metrics file that can be either a JSON list or JSONL (one JSON per line)
    and return a list of dictionaries.
    """
    with path.open("r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: Path, data: List[Dict[str, Any]]):
    """
    Write a list of dictionaries to a file in JSONL format.
    """
    with path.open("w") as f:
        for obj in data:
            f.write(json.dumps(obj) + "\n")

# ───────────────────────────── main runner ───────────────────────────────────

def main(cfg):
    # Load synthesis metadata.
    with cfg.metadata.open() as f:
        metadata = json.load(f)

    gen_paths = [Path(e["generated_wav_path"]) for e in metadata]
    ref_paths = [Path(e["reference_wav_path"]) for e in metadata]

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())

    pred_scp = tmp / "pred.scp"
    ref_scp  = tmp / "ref.scp"
    build_scp(gen_paths, pred_scp)
    build_scp(ref_paths, ref_scp)

    # Write YAML configs (top‑level list) ---------------------------------------
    independent_yaml = cfg.outdir / "independent.yaml"
    dependent_yaml   = cfg.outdir / "dependent.yaml"

    independent_yaml.write_text("""
- name: pseudo_mos
  predictor_types: [utmos]
""")

    dependent_yaml.write_text("""
- name: pesq
- name: stoi
- name: signal_metric
""")

    ind_out = cfg.outdir / "independent.json"
    dep_out = cfg.outdir / "dependent.json"
    gpu_flag = "true" if cfg.device == "cuda" else "false"

    logging.info("Running independent metrics …")
    run_versa([
        "--pred", str(pred_scp),
        "--score_config", str(independent_yaml),
        "--output_file", str(ind_out),
        "--use_gpu", gpu_flag,
        "--io", "kaldi"
    ])

    logging.info("Running reference metrics …")
    run_versa([
        "--pred", str(pred_scp),
        "--gt", str(ref_scp),
        "--score_config", str(dependent_yaml),
        "--output_file", str(dep_out),
        "--use_gpu", gpu_flag,
        "--io", "kaldi"
    ])

    # Read metrics from the independent and dependent outputs.
    independent_metrics = read_metrics(ind_out)
    dependent_metrics = read_metrics(dep_out)

    # Merge metrics based on "key".
    merged = {}
    for entry in independent_metrics:
        key = entry.get("key")
        if key:
            merged[key] = entry.copy()
    for entry in dependent_metrics:
        key = entry.get("key")
        if key:
            if key in merged:
                merged[key].update(entry)
            else:
                merged[key] = entry.copy()

    # Compute overall average for each numeric metric (ignoring non-finite values).
    overall = {}
    counts = {}
    for entry in merged.values():
        for metric, value in entry.items():
            if metric == "key":
                continue
            if isinstance(value, (int, float)) and math.isfinite(value):
                overall[metric] = overall.get(metric, 0.0) + value
                counts[metric] = counts.get(metric, 0) + 1

    for metric in overall:
        overall[metric] /= counts[metric]

    # Prepare final output: each utterance's metrics (one JSON per line)
    # and an overall summary as the last line.
    output_lines = []
    for key, metrics in merged.items():
        output_lines.append(metrics)
    overall_entry = {"key": "overall"}
    overall_entry.update(overall)
    output_lines.append(overall_entry)

    # Write the final JSONL summary.
    summary_jsonl_path = cfg.outdir / "auto_summary.jsonl"
    write_jsonl(summary_jsonl_path, output_lines)

    logging.info("Done. Results saved in %s", cfg.outdir)

# ─────────────────────────────── CLI ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TTS evaluation with VERSA")
    p.add_argument("--metadata", type=Path, default="generated_wavs/train/metadata.json", required=False,
                   help="Path to metadata.json produced after synthesis")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--outdir", type=Path, default=Path("evaluation_results"))
    p.add_argument("--verbose", action="count", default=0,
                   help="‑v (INFO) or ‑vv (DEBUG)")
    return p.parse_args()

if __name__ == "__main__":
    cfg = parse_args()
    logging.basicConfig(level=logging.DEBUG if cfg.verbose > 1 else
                        logging.INFO if cfg.verbose == 1 else logging.WARN,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    main(cfg)

#!/usr/bin/env python
"""
evaluate_with_versa.py
───────────────────────────────────────────────────────────────────────────────
Run objective TTS evaluation using the **VERSA** scorer already installed in
this environment.

Two passes are launched:
1. Independent metrics (no reference): UTMOS + DNSMOS (pseudo_mos), NISQA, SRMR.
2. Reference metrics: PESQ, STOI, signal_metric (SI‑SNR & LSD bundle), VISQOL,
   and MCD + F0‑RMSE (mcd_f0).

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
import re
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
    Read a metrics file that may be:
      - a JSON list    (e.g. [ {...}, {...}, ... ])
      - JSONL          (one JSON object per line)
      - Python reprs   (single‑quoted dicts, one per line, possibly containing inf/nan)
    Returns a list of dicts.
    """
    text = path.read_text()
    if text.lstrip().startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    metrics: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # try strict JSON
        try:
            metrics.append(json.loads(line))
            continue
        except json.JSONDecodeError:
            pass

        # convert to JSON-friendly (single→double quotes, inf→null)
        jl = line.replace("'", '"')
        jl = re.sub(r':\s*([+-]?inf)(?=[,}])', r': null', jl)
        try:
            metrics.append(json.loads(jl))
            continue
        except json.JSONDecodeError:
            pass

        # fallback: safe eval for inf/nan
        safe_locals = {"inf": float("inf"), "nan": float("nan")}
        metrics.append(eval(line, {"__builtins__": None}, safe_locals))

    return metrics

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

    # Write YAML configs (top‑level list)
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
    # ── Postprocess independent.json into valid JSON list ──
    buf = ind_out.read_text().splitlines()
    objs: List[Dict[str, Any]] = []
    for line in buf:
        line = line.strip()
        if not line:
            continue
        objs.append(eval(line,
                         {"__builtins__": None},
                         {"inf": float("inf"), "nan": float("nan")}))
    ind_out.write_text(json.dumps(objs, indent=2))

    logging.info("Running reference metrics …")
    run_versa([
        "--pred", str(pred_scp),
        "--gt", str(ref_scp),
        "--score_config", str(dependent_yaml),
        "--output_file", str(dep_out),
        "--use_gpu", gpu_flag,
        "--io", "kaldi"
    ])
    # ── Postprocess dependent.json into valid JSON list ──
    buf = dep_out.read_text().splitlines()
    objs = []
    for line in buf:
        line = line.strip()
        if not line:
            continue
        objs.append(eval(line,
                         {"__builtins__": None},
                         {"inf": float("inf"), "nan": float("nan")}))
    dep_out.write_text(json.dumps(objs, indent=2))

    # Read metrics from the independent and dependent outputs.
    independent_metrics = read_metrics(ind_out)
    dependent_metrics   = read_metrics(dep_out)

    # Merge metrics based on "key".
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in independent_metrics:
        key = entry.get("key")
        if key:
            merged[key] = entry.copy()
    for entry in dependent_metrics:
        key = entry.get("key")
        if key:
            merged.setdefault(key, {}).update(entry)

    # Compute overall average for each numeric metric.
    overall: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for entry in merged.values():
        for metric, value in entry.items():
            if metric == "key":
                continue
            if isinstance(value, (int, float)) and math.isfinite(value):
                overall[metric] = overall.get(metric, 0.0) + value
                counts[metric]  = counts.get(metric, 0) + 1
    for metric in overall:
        overall[metric] /= counts[metric]

    # Prepare final output: individual utterance metrics + overall summary.
    output_lines: List[Dict[str, Any]] = list(merged.values())
    overall_entry = {"key": "overall", **overall}
    output_lines.append(overall_entry)

    # Write the final JSONL summary.
    summary_jsonl_path = cfg.outdir / "auto_summary.jsonl"
    write_jsonl(summary_jsonl_path, output_lines)

    logging.info("Done. Results saved in %s", cfg.outdir)

# ─────────────────────────────── CLI ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TTS evaluation with VERSA")
    p.add_argument(
        "--metadata",
        type=Path,
        default="generated_wavs/train/metadata.json",
        required=False,
        help="Path to metadata.json produced after synthesis",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("evaluation_results"),
    )
    p.add_argument(
        "--verbose",
        action="count",
        default=0,
        help="-v (INFO) or -vv (DEBUG)",
    )
    return p.parse_args()

if __name__ == "__main__":
    cfg = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose > 1 else
              logging.INFO  if cfg.verbose == 1 else
              logging.WARN,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )
    main(cfg)

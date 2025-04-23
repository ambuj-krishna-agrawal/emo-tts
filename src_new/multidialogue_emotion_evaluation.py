#!/usr/bin/env python3
"""
emotion_evaluation.py
-------------------------------------------------------------------------------
Evaluates how well an LLM‑TTS pipeline matches emotions at two stages
and saves every evaluation input to JSONL for reproducibility.

Final paths that should exist:
- OUTPUT_DIR/[model_name]_eval_inputs.jsonl
- OUTPUT_DIR/[model_name]_emotion2vec_raw_results.csv (or .jsonl if pandas not available)
- OUTPUT_DIR/[model_name]_stats.json
- OUTPUT_DIR/all_models_stats.json
"""

from __future__ import annotations
import argparse, json, logging, sys, shutil, tempfile, subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
try:                       # tolerate a broken pandas install
    import pandas as pd
except Exception:
    pd = None

from tqdm import tqdm      # progress bars

# -----------------------------------------------------------------------------#
#  Paths - EDIT THESE                                                           #
# -----------------------------------------------------------------------------#
BASE_DIR = Path("/home/ambuja/emo-tts")
DATA_DIR = Path("/data/group_data/starlight/gpa/tts")
PAIRS_METADATA_PATH = DATA_DIR / "multidialog_sds_pairs/pairs_metadata.json"

# Dictionary of model directories to evaluate
MODEL_DIRS = {
    "llama_3_70b_q4": DATA_DIR / "multidialogue_emotivoice_out/llama_3_70b_q4",
    "mistral_7b_q4": DATA_DIR / "multidialogue_emotivoice_out/mistral_7b_q4",
    "llama_3_3b_q4": DATA_DIR / "multidialogue_emotivoice_out/llama_3_3b_q4"
}

EMOTION_PLANNING_DIR = DATA_DIR / "multidialog_emotion_planning/llama_3_3b_q4"
OUTPUT_DIR = DATA_DIR / "multidialogue_evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------#
#  Emotion mapping                                                              #
# -----------------------------------------------------------------------------#
EMOTION_MAPPING: Dict[str, int] = {
    "angry": 0, "anger": 0, "frustrated": 0,
    "disgust": 1, "disgusted": 1,
    "fear": 2, "fearful": 2,
    "happy": 3, "joy": 3, "excited": 3, "curious to dive deeper": 3,
    "neutral": 4,
    "other": 5, "none": 5,
    "sad": 6, "sadness": 6, "bored": 6,
    "surprise": 7, "surprised": 7, "curious": 7,
}
DEFAULT_EMOTIONS = sorted(EMOTION_MAPPING)

# -----------------------------------------------------------------------------#
#  Logging                                                                      #
# -----------------------------------------------------------------------------#
def setup_logger(verbosity: int) -> logging.Logger:
    logger = logging.getLogger("emotion_eval")
    if logger.handlers:
        return logger
    level = logging.WARNING if verbosity == 0 else (
            logging.INFO if verbosity == 1 else logging.DEBUG)
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger

# -----------------------------------------------------------------------------#
#  Helpers                                                                      #
# -----------------------------------------------------------------------------#
def load_json(path: Path, logger: logging.Logger) -> Any:
    logger.info("Loading %s", path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return json.loads(path.read_text())

def dump_jsonl(records: List[dict], path: Path):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def dump_json(data: dict, path: Path):
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# -----------------------------------------------------------------------------#
#  Part A: Evaluate emotion choice accuracy                                     #
# -----------------------------------------------------------------------------#
def eval_emotion_choice(pairs_meta, planning_path, lg):
    data = load_json(planning_path, lg)
    pair2emo = {}
    
    for d in data:
        if "pair_id" in d and "reference_emotion" in d and "standardized_reference_emotion" in d:
            # Use standardized reference emotion if available, otherwise use reference_emotion
            ref_emotion = d.get("standardized_reference_emotion", d["reference_emotion"]).lower()
            target_emotion = d.get("target_emotion", "").lower()
            pair2emo[d["pair_id"]] = (ref_emotion, target_emotion)

    tot = match = 0
    for ref, pred in pair2emo.values():
        if ref not in EMOTION_MAPPING or pred not in EMOTION_MAPPING:
            continue
        tot += 1
        match += (ref == pred)
    acc = match / tot if tot else 0
    lg.info("LLM emotion‑choice accuracy: %.2f%% (%d/%d)", acc*100, match, tot)
    return acc, {"accuracy": acc, "matched": match, "total": tot}

# -----------------------------------------------------------------------------#
#  Emotion2vec wrappers                                                         #
# -----------------------------------------------------------------------------#
def run_emotion2vec_on_audio(audio_paths: Dict[str, Path],
                             output_dir: Path,
                             lg: logging.Logger,
                             prefer_versa: bool = True,
                             num_threads: int = 4) -> Dict[str, Dict[str, Any]]:
    if prefer_versa:
        try:
            return _run_with_versa(audio_paths, output_dir, lg)
        except (ImportError, ModuleNotFoundError):
            lg.warning("VERSA not available – falling back to FunASR.")
    return _run_with_funasr(audio_paths, output_dir, lg, num_threads)

# -- VERSA --------------------------------------------------------------------#
def _run_with_versa(audio_paths, out_dir, lg):
    lg.info("Running emotion2vec via VERSA")
    tmp = Path(tempfile.mkdtemp())
    scp = tmp / "audio.scp"
    with scp.open("w") as f:
        for k, p in audio_paths.items():
            f.write(f"{k} {p}\n")
    (tmp / "emotion.yaml").write_text(
        "- name: emotion2vec\n  predictor_types: [emotion2vec]\n"
        "  params: {model_name: iic/emotion2vec_plus_large}\n")

    out_json = tmp / "versa.json"
    cmd = [sys.executable, "-m", "versa.bin.scorer",
           "--pred", scp, "--score_config", tmp/"emotion.yaml",
           "--output_file", out_json, "--use_gpu", "true", "--io", "kaldi"]
    subprocess.run(cmd, check=True)

    res = {}
    for line in out_json.read_text().splitlines():
        d = eval(line, {"__builtins__": None}, {"inf": float("inf"), "nan": float("nan")})
        scores, labels = d["emotion_scores"], d["emotion_labels"]
        idx = int(np.argmax(scores))
        res[d["key"]] = {
            "scores": scores, "labels": labels,
            "predicted_emotion_idx": idx,
            "predicted_emotion_score": float(scores[idx]),
            "predicted_emotion_label": labels[idx],
        }
    shutil.rmtree(tmp, ignore_errors=True)
    return res

# -- FunASR with multithreading -----------------------------------------------#
def _process_audio_with_funasr(key, wav, model, lg):
    """Process a single audio file with FunASR - for multithreading"""
    if not wav.exists():
        lg.warning("Missing %s", wav)
        return key, None
    
    try:
        rec = model.generate(str(wav), granularity="utterance", extract_embedding=False)
    except Exception as exc:
        lg.error("FunASR failed for %s: %s", key, exc)
        return key, None

    # normalise output
    if isinstance(rec, list) and rec and isinstance(rec[0], dict):
        rec = rec[0]
    scores, labels = rec.get("scores") or rec.get("score"), rec.get("labels") or rec.get("label")
    if scores is None or labels is None:
        lg.error("Bad output for %s", key)
        return key, None
    
    idx = int(np.argmax(scores))
    return key, {
        "scores": scores, "labels": labels,
        "predicted_emotion_idx": idx,
        "predicted_emotion_score": float(scores[idx]),
        "predicted_emotion_label": labels[idx],
    }

def _run_with_funasr(audio_paths, out_dir, lg, num_threads=4):
    try:
        from funasr import AutoModel
    except ImportError as exc:
        lg.error("FunASR missing: %s", exc)
        sys.exit(1)

    try:
        model = AutoModel(model="iic/emotion2vec_plus_large")
    except Exception:
        lg.info("Falling back to base_finetuned model")
        model = AutoModel(model="iic/emotion2vec_base_finetuned")

    res = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(_process_audio_with_funasr, key, wav, model, lg): key 
                   for key, wav in audio_paths.items()}
        
        with tqdm(total=len(audio_paths), desc="Processing audio files", leave=False) as pbar:
            for future in as_completed(futures):
                key, result = future.result()
                if result is not None:
                    res[key] = result
                pbar.update(1)
    
    return res

# -----------------------------------------------------------------------------#
#  Part B: Evaluate speech emotion accuracy                                     #
# -----------------------------------------------------------------------------#
def eval_speech_accuracy(model_name: str,
                         pairs_meta, llm_meta,
                         planning_data,
                         emo2vec_res, audio_paths,
                         out_dir: Path, lg):

    # Build mappings from planning data
    plan_map = {}
    for d in planning_data:
        if "pair_id" in d:
            ref_emotion = d.get("standardized_reference_emotion", d.get("reference_emotion", "")).lower()
            target_emotion = d.get("target_emotion", "").lower()
            plan_map[d["pair_id"]] = (ref_emotion, target_emotion)

    # Get emotion mappings
    gold = {p["pair_id"]: EMOTION_MAPPING.get(p["emotion_B"].lower())
            for p in pairs_meta if "emotion_B" in p}
    neutral = {i["pair_id"]: EMOTION_MAPPING["neutral"]
               for i in llm_meta if i.get("emotion", "").lower() == "neutral"}
    steered = {i["pair_id"]: EMOTION_MAPPING.get(plan_map[i["pair_id"]][1], None)
              for i in llm_meta if i.get("emotion", "").lower() == "steered"
              and i["pair_id"] in plan_map and plan_map[i["pair_id"]][1] in EMOTION_MAPPING}

    stats = {k: {"c": 0, "t": 0} for k in
             ("gold", "neutral", "steered_ref", "steered_model")}

    dump = []
    for pid in gold.keys() & neutral.keys() & steered.keys():
        ref_idx, mod_idx = gold[pid], steered[pid]
        pred_g = emo2vec_res.get(f"gold_{pid}", {}).get("predicted_emotion_idx")
        pred_n = emo2vec_res.get(f"neutral_{pid}", {}).get("predicted_emotion_idx")
        pred_s = emo2vec_res.get(f"steered_{pid}", {}).get("predicted_emotion_idx")

        dump.append({
            "pair_id": pid,
            "gold_audio": str(audio_paths.get(f"gold_{pid}", "")),
            "neutral_audio": str(audio_paths.get(f"neutral_{pid}", "")),
            "steered_audio": str(audio_paths.get(f"steered_{pid}", "")),
            "ref_emotion_idx": ref_idx,
            "model_emotion_idx": mod_idx,
            "pred_gold_idx": pred_g,
            "pred_neutral_idx": pred_n,
            "pred_steered_idx": pred_s,
        })

        if pred_g is not None:
            stats["gold"]["t"] += 1; stats["gold"]["c"] += (pred_g == ref_idx)
        if pred_n is not None:
            stats["neutral"]["t"] += 1; stats["neutral"]["c"] += (pred_n == ref_idx)
        if pred_s is not None:
            stats["steered_ref"]["t"] += 1; stats["steered_ref"]["c"] += (pred_s == ref_idx)
            stats["steered_model"]["t"] += 1; stats["steered_model"]["c"] += (pred_s == mod_idx)

    dump_jsonl(dump, out_dir / f"{model_name}_eval_inputs.jsonl")
    lg.info("Saved per‑item JSONL for %s", model_name)

    # raw results
    if pd is not None:
        pd.DataFrame.from_dict(emo2vec_res, orient="index").to_csv(
            out_dir / f"{model_name}_emotion2vec_raw_results.csv")
    else:
        dump_jsonl([{"key": k, **v} for k, v in emo2vec_res.items()],
                  out_dir / f"{model_name}_emotion2vec_raw_results.jsonl")

    # Calculate and format accuracies
    results = {}
    for k, d in stats.items():
        acc = d["c"]/d["t"] if d["t"] else 0
        lg.info("%-14s: %.2f%% (%d/%d)", k, acc*100, d["c"], d["t"])
        results[k] = {"accuracy": acc, "correct": d["c"], "total": d["t"]}
    
    return results

# -----------------------------------------------------------------------------#
#  Main                                                                         #
# -----------------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="count", default=0)
    ap.add_argument("--no-versa", action="store_true",
                    help="Force FunASR even if VERSA is present")
    ap.add_argument("--threads", type=int, default=4,
                    help="Number of threads for parallel processing with FunASR")
    args = ap.parse_args()

    lg = setup_logger(args.verbose)
    lg.info("=== Emotion evaluation pipeline ===")

    pairs_meta = load_json(PAIRS_METADATA_PATH, lg)

    partA_summary = {}
    all_model_stats = {}

    for model_name, model_dir in MODEL_DIRS.items():
        lg.info("--- %s ---", model_name)

        plan_file = EMOTION_PLANNING_DIR / "results.jsonl"
        tts_meta = model_dir / "metadata.jsonl"
        if not plan_file.exists() or not tts_meta.exists():
            lg.error("Missing planning or metadata for %s – skipping", model_name)
            continue

        accA, partA_stats = eval_emotion_choice(pairs_meta, plan_file, lg)
        partA_summary[model_name] = partA_stats

        llm_meta = load_json(tts_meta, lg)

        # Build audio path dict
        audio_paths: Dict[str, Path] = {}
        with tqdm(total=len(pairs_meta), desc="Processing gold audio paths", leave=False) as pbar:
            for p in pairs_meta:
                pid = p["pair_id"]; wav = Path(p["audio_path_B"])
                if not wav.is_absolute(): wav = DATA_DIR / p["audio_path_B"]
                audio_paths[f"gold_{pid}"] = wav
                pbar.update(1)
        
        with tqdm(total=len(llm_meta), desc="Processing model audio paths", leave=False) as pbar:
            for rec in llm_meta:
                pid, emo = rec["pair_id"], rec["emotion"].lower()
                audio_paths[f"{emo}_{pid}"] = Path(rec["wav"])
                pbar.update(1)

        emo2vec_res = run_emotion2vec_on_audio(
            audio_paths, OUTPUT_DIR, lg, 
            prefer_versa=not args.no_versa,
            num_threads=args.threads)

        partB_stats = eval_speech_accuracy(model_name,
                          pairs_meta, llm_meta,
                          load_json(plan_file, lg),
                          emo2vec_res, audio_paths,
                          OUTPUT_DIR, lg)
        
        # Store all stats for this model
        all_model_stats[model_name] = {
            "part_A": partA_stats,
            "part_B": partB_stats
        }
        
        # Save model-specific stats
        dump_json(all_model_stats[model_name], 
                 OUTPUT_DIR / f"{model_name}_stats.json")

    # Save combined stats for all models
    dump_json({
        "part_A_summary": partA_summary,
        "all_model_stats": all_model_stats
    }, OUTPUT_DIR / "all_models_stats.json")

    lg.info("\n=== Part A summary ===")
    for name, stats in partA_summary.items():
        lg.info("%-10s: %.2f%% (%d/%d)", name, 
                stats["accuracy"]*100, stats["matched"], stats["total"])

if __name__ == "__main__":
    main()
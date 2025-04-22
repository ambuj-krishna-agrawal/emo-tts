#!/usr/bin/env python3
"""
tts_batch_offline_emotivoice.py
--------------------------------
• Converts every planning record in  *vllm_emotion_planning_results/*
  into a single test‑file that   **inference_am_vocoder_joint.py** expects.
• Invokes   inference_am_vocoder_joint.py   once (GPU‑friendly).
• Renames / copies the generated WAVs into
      emotivoice_out/<planning_model>/<pair_id>_{neutral|steered}.wav
  and writes per‑model   metadata.jsonl, exactly like the old API script.

Dependencies: the standard EmotiVoice env (`conda activate EmotiVoice`)
and the pretrained checkpoints under  outputs/prompt_tts_open_source_joint/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from tqdm import tqdm

# Set up logging
log_file = Path("emotivoice_batch_process.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- Set up the paths -------------------------
# Path to the EmotiVoice installation
EMOTIVOICE_PATH = Path("/home/ambuja/emo-tts/EmotiVoice")
# Path to the EmotiVoice models
EMOTIVOICE_MODELS_PATH = Path("/data/group_data/starlight/gpa/tts/EmotiVoice_models")
# Add EmotiVoice to the path
sys.path.insert(0, str(EMOTIVOICE_PATH))

class ProcessTimer:
    """Timer class to track progress and estimate remaining time."""
    def __init__(self, total_steps: int, description: str):
        self.total_steps = total_steps
        self.description = description
        self.start_time = time.time()
        self.step_times = []
        self.current_step = 0
        self.pbar = tqdm(total=total_steps, desc=description)
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps and recalculate estimates."""
        now = time.time()
        if self.current_step > 0:
            self.step_times.append(now - self.start_time - sum(self.step_times))
        self.current_step += n
        self.pbar.update(n)
        
        # Update description with time estimate
        if len(self.step_times) > 0:
            avg_time = sum(self.step_times) / len(self.step_times)
            eta = avg_time * (self.total_steps - self.current_step)
            eta_str = str(timedelta(seconds=int(eta)))
            self.pbar.set_description(f"{self.description} (ETA: {eta_str})")
    
    def close(self) -> None:
        """Close the progress bar and report total time."""
        self.pbar.close()
        elapsed = time.time() - self.start_time
        logger.info(f"Completed {self.description} in {timedelta(seconds=int(elapsed))}")

def validate_paths() -> bool:
    """Validate that all required paths exist and are accessible."""
    paths_to_check = [
        (EMOTIVOICE_PATH, "EmotiVoice installation"),
        (EMOTIVOICE_MODELS_PATH, "EmotiVoice models"),
        (JSONL_DIR, "JSONL directory")
    ]
    
    all_valid = True
    for path, description in paths_to_check:
        if not path.exists():
            logger.error(f"Path not found: {path} ({description})")
            all_valid = False
    
    return all_valid

# ---------- EmotiVoice frontend (phoneme converter) -------------------------
try:
    from frontend import g2p_cn_en, read_lexicon, G2p  # part of EmotiVoice repo
    logger.info("Successfully imported EmotiVoice frontend modules")
except ImportError as e:
    logger.error(f"Failed to import EmotiVoice modules: {e}")
    logger.error("Make sure you've activated the EmotiVoice environment (conda activate EmotiVoice)")
    sys.exit(1)

# ------------------------- Paths & constants --------------------------------
BASE_DIR     = Path("/home/ambuja/emo-tts")  # Your working directory
DATA_DIR = Path("/data/group_data/starlight/gpa/tts")
JSONL_DIR    = DATA_DIR / "multidialog_emotion_planning_1"
OUT_ROOT     = DATA_DIR / "multidialogue_emotivoice_out"
TMP_DIR      = BASE_DIR / "_ev_tmp"
TMP_DIR.mkdir(exist_ok=True)

TEST_FILE    = TMP_DIR / "batch_for_tts.txt"      # input for inference script
SPEAKER_ID   = os.getenv("EMOTIVOICE_SPK", "8051")  # any ID in speaker2id.txt
CHECKPOINT   = "g_00140000"                       # change if you like
LOGDIR       = "prompt_tts_open_source_joint"     # official default
CONFIG_DIR   = "config/joint"  # Use relative path for EmotiVoice script
EMO_NEUTRAL  = "Neutral"                          # baseline prompt

# Status file to track progress
STATUS_FILE = TMP_DIR / "batch_progress.json"

def update_status(stage: str, progress: int, total: int, eta: Optional[float] = None) -> None:
    """Update the status file with current progress."""
    status = {
        "stage": stage,
        "progress": progress,
        "total": total,
        "percent_complete": round((progress / total) * 100, 2) if total > 0 else 0,
        "start_time": getattr(update_status, "start_time", time.time()),
        "current_time": time.time(),
    }
    
    if eta is not None:
        status["estimated_completion"] = time.time() + eta
        status["eta_formatted"] = str(timedelta(seconds=int(eta)))
    
    elapsed = status["current_time"] - status["start_time"]
    status["elapsed_formatted"] = str(timedelta(seconds=int(elapsed)))
    
    # Calculate ETA based on elapsed time if not provided
    if eta is None and progress > 0:
        eta_sec = (elapsed / progress) * (total - progress)
        status["estimated_completion"] = time.time() + eta_sec
        status["eta_formatted"] = str(timedelta(seconds=int(eta_sec)))
    
    with STATUS_FILE.open("w") as f:
        json.dump(status, f, indent=2)

# Store start time for the first call to update_status
update_status.start_time = time.time()

def main():
    # Validate paths before proceeding
    if not validate_paths():
        logger.error("Path validation failed. Please check the paths and try again.")
        sys.exit(1)
    
    logger.info(f"Starting EmotiVoice batch processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Working directory: {BASE_DIR}")
    logger.info(f"Output directory: {OUT_ROOT}")
    
    # ----------------------------- Load records ---------------------------------
    jsonl_paths = sorted(JSONL_DIR.glob("*_results.jsonl"))
    if not jsonl_paths:
        logger.error(f"No jsonl files found under {JSONL_DIR}")
        sys.exit(1)
    
    logger.info(f"Found {len(jsonl_paths)} JSONL files to process")
    
    records: List[Dict] = []
    load_timer = ProcessTimer(len(jsonl_paths), "Loading JSONL files")
    
    for p in jsonl_paths:
        planning_model = p.stem
        try:
            with p.open() as f:
                file_records = [json.loads(line) for line in f if line.strip()]
                for rec in file_records:
                    rec["planning_model"] = planning_model
                    records.append(rec)
            load_timer.update()
            logger.info(f"Loaded {len(file_records)} records from {p.name}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing {p}: {e}")
            load_timer.update()
    
    load_timer.close()
    logger.info(f"Total records loaded: {len(records)}")
    update_status("loading_records", len(records), len(records))
    
    # ------------------------- Build test‑file lines ----------------------------
    logger.info("Loading lexicon and g2p models...")
    try:
        lexicon_path = f"{EMOTIVOICE_PATH}/lexicon/librispeech-lexicon.txt"
        lexicon = read_lexicon(lexicon_path)
        g2p = G2p()
        logger.info("Lexicon and g2p models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load lexicon or g2p: {e}")
        sys.exit(1)
    
    lines: List[str] = []
    index2meta: List[Tuple[str, str, str]] = []   # (pair_id, model_name, tag)
    
    def to_phoneme(txt: str) -> str:
        """English g2p → token string understood by EmotiVoice."""
        return g2p_cn_en(txt, g2p, lexicon)       # returns e.g. "<sos/eos> [IH0] … <sos/eos>"
    
    logger.info("Converting text to phonemes...")
    phoneme_timer = ProcessTimer(len(records) * 2, "Converting text to phonemes")
    update_status("converting_to_phonemes", 0, len(records) * 2)
    
    for i, rec in enumerate(records):
        pid   = rec["pair_id"]
        model = rec["planning_model"]
        
        try:
            # baseline (neutral)
            text_b = rec["baseline_reply"].strip()
            ph_b   = to_phoneme(text_b)
            lines.append(f"{SPEAKER_ID}|{EMO_NEUTRAL}|{ph_b}|{text_b}")
            index2meta.append((pid, model, "neutral"))
            phoneme_timer.update()
            update_status("converting_to_phonemes", i*2+1, len(records)*2)
            
            # emotion‑steered
            emo    = rec["target_emotion"].capitalize()
            text_s = rec["emotion_steered_reply"].strip()
            ph_s   = to_phoneme(text_s)
            lines.append(f"{SPEAKER_ID}|{emo}|{ph_s}|{text_s}")
            index2meta.append((pid, model, "steered"))
            phoneme_timer.update()
            update_status("converting_to_phonemes", i*2+2, len(records)*2)
        except Exception as e:
            logger.error(f"Error processing record {pid}: {e}")
    
    phoneme_timer.close()
    
    # Write the test file
    try:
        TEST_FILE.write_text("\n".join(lines), encoding="utf‑8")
        logger.info(f"Wrote {len(lines)} lines to {TEST_FILE}")
    except Exception as e:
        logger.error(f"Failed to write test file: {e}")
        sys.exit(1)
    
    # ------------------------ Run official inference ---------------------------
    logger.info("Preparing for inference...")
    update_status("preparing_inference", 0, 1)
    
    # Create symbolic link to the models directory if needed
    outputs_dir = EMOTIVOICE_PATH / "outputs"
    if not outputs_dir.exists():
        logger.info(f"Creating symbolic link from {EMOTIVOICE_MODELS_PATH / 'outputs'} to {outputs_dir}")
        try:
            os.symlink(EMOTIVOICE_MODELS_PATH / 'outputs', outputs_dir)
        except Exception as e:
            logger.error(f"Failed to create symbolic link: {e}")
            sys.exit(1)
    
    update_status("preparing_inference", 1, 1)
    
    # Create inference script
    script_path = TMP_DIR / "run_inference.sh"
    script_content = f"""#!/bin/bash
cd {EMOTIVOICE_PATH}

# Check if symlink exists
if [ ! -d "outputs" ]; then
    echo "Creating symbolic link to model outputs"
    ln -sf {EMOTIVOICE_MODELS_PATH}/outputs outputs
fi

# Verify checkpoint path exists
if [ ! -d "outputs/{LOGDIR}/ckpt" ]; then
    echo "ERROR: Checkpoint path does not exist: outputs/{LOGDIR}/ckpt"
    echo "Available directories in outputs:"
    ls -l outputs/
    echo "Available directories in outputs/{LOGDIR} (if it exists):"
    [ -d "outputs/{LOGDIR}" ] && ls -l outputs/{LOGDIR}/
    exit 1
fi

# Count total steps for progress reporting
total_lines=$(wc -l < {TEST_FILE})
echo "Total lines to process: $total_lines"
echo "Start time: $(date)"

# Run inference with progress monitoring
python inference_am_vocoder_joint.py \\
    --logdir {LOGDIR} \\
    --config_folder {CONFIG_DIR} \\
    --checkpoint {CHECKPOINT} \\
    --test_file {TEST_FILE} \\
    2>&1 | tee {TMP_DIR}/inference.log

# Capture timestamp at end
echo "End time: $(date)"
"""
    
    try:
        script_path.write_text(script_content)
        os.chmod(script_path, 0o755)  # Make executable
        logger.info(f"Created inference script at {script_path}")
    except Exception as e:
        logger.error(f"Failed to create inference script: {e}")
        sys.exit(1)
    
    # Run inference
    logger.info("Starting inference process...")
    update_status("running_inference", 0, len(lines))
    
    inference_start = time.time()
    
    try:
        # Launch process and connect to its output for progress monitoring
        process = subprocess.Popen(
            str(script_path),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Read output lines to monitor progress
        progress_count = 0
        for line in process.stdout:
            print(line.rstrip())  # Echo to console
            
            # Try to detect progress
            if "Processing" in line and "of" in line:
                try:
                    parts = line.strip().split()
                    progress_idx = parts.index("Processing") + 1
                    progress_count = int(parts[progress_idx])
                    update_status("running_inference", progress_count, len(lines))
                except (ValueError, IndexError):
                    pass
        
        # Wait for process to complete
        return_code = process.wait()
        if return_code != 0:
            logger.error(f"Inference process exited with code {return_code}")
            sys.exit(return_code)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        sys.exit(1)
    
    inference_time = time.time() - inference_start
    logger.info(f"Inference completed in {timedelta(seconds=int(inference_time))}")
    update_status("running_inference", len(lines), len(lines))
    
    # ----------------------- Collect & rename WAVs ------------------------------
    logger.info("Collecting and renaming WAV files...")
    update_status("collecting_wavs", 0, len(index2meta))
    
    root_path = EMOTIVOICE_MODELS_PATH / "outputs" / LOGDIR
    wav_dir = root_path / "test_audio" / "audio" / CHECKPOINT
    
    try:
        wavs = sorted(wav_dir.glob("*.wav"), key=lambda p: int(p.stem))
    except Exception as e:
        logger.error(f"Failed to collect WAV files: {e}")
        sys.exit(1)
    
    if len(wavs) != len(index2meta):
        logger.error(f"Mismatch between WAV count ({len(wavs)}) and metadata list ({len(index2meta)}).")
        sys.exit(1)
    
    logger.info(f"Found {len(wavs)} WAV files to process")
    
    meta_per_model: Dict[str, List[Dict]] = {}
    wav_timer = ProcessTimer(len(wavs), "Processing WAV files")
    
    for i, (wav_path, (pid, model, tag)) in enumerate(zip(wavs, index2meta)):
        try:
            tgt_dir = OUT_ROOT / model
            tgt_dir.mkdir(parents=True, exist_ok=True)
            tgt_file = tgt_dir / f"{pid}_{tag}.wav"
            shutil.copy2(wav_path, tgt_file)
            
            # Add reference_emotion to metadata
            ref_emotion = next((rec["reference_emotion"] for rec in records if rec["pair_id"] == pid), None)
            
            meta_per_model.setdefault(model, []).append({
                "pair_id": pid,
                "wav": str(tgt_file),
                "emotion": tag,
                "reference_emotion": ref_emotion,
                "orig_path": str(wav_path)
            })
            
            wav_timer.update()
            update_status("collecting_wavs", i+1, len(wavs))
        except Exception as e:
            logger.error(f"Error processing WAV {wav_path}: {e}")
    
    wav_timer.close()
    
    # Write metadata files
    logger.info("Writing metadata files...")
    update_status("writing_metadata", 0, len(meta_per_model))
    
    for i, (model, entries) in enumerate(meta_per_model.items()):
        try:
            meta_p = OUT_ROOT / model / "metadata.jsonl"
            with meta_p.open("w") as f:
                for m in entries:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            logger.info(f"[{model}] Wrote {len(entries)} entries to {meta_p}")
            update_status("writing_metadata", i+1, len(meta_per_model))
        except Exception as e:
            logger.error(f"Error writing metadata for {model}: {e}")
    
    # Final status update
    total_time = time.time() - update_status.start_time
    logger.info(f"Total processing time: {timedelta(seconds=int(total_time))}")
    logger.info(f"All audio & metadata are under {OUT_ROOT}")
    
    # Mark completion in status file
    with STATUS_FILE.open("w") as f:
        json.dump({
            "stage": "complete",
            "total_time": total_time,
            "total_time_formatted": str(timedelta(seconds=int(total_time))),
            "completion_time": datetime.now().isoformat(),
            "records_processed": len(records),
            "wavs_generated": len(wavs),
            "output_directory": str(OUT_ROOT)
        }, f, indent=2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
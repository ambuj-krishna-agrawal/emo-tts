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

import json, os, shutil, subprocess, sys
from pathlib import Path
from typing import List, Tuple, Dict

# ---------- Set up the paths -------------------------
# Path to the EmotiVoice installation
EMOTIVOICE_PATH = Path("/home/ambuja/emo-tts/EmotiVoice")
# Path to the EmotiVoice models
EMOTIVOICE_MODELS_PATH = Path("/data/group_data/starlight/gpa/tts/EmotiVoice_models")
# Add EmotiVoice to the path
sys.path.insert(0, str(EMOTIVOICE_PATH))

# ---------- EmotiVoice frontend (phoneme converter) -------------------------
from frontend import g2p_cn_en, read_lexicon, G2p  # part of EmotiVoice repo

# ------------------------- Paths & constants --------------------------------
BASE_DIR     = Path("/home/ambuja/emo-tts")  # Your working directory
# JSONL_DIR    = BASE_DIR / "vllm_emotion_planning_results"
JSONL_DIR    = BASE_DIR / "emotion_planning_results_custom_balanced"
OUT_ROOT     = BASE_DIR / "emotivoice_out_custom_balanced"
TMP_DIR      = BASE_DIR / "_ev_tmp"
TMP_DIR.mkdir(exist_ok=True)

TEST_FILE    = TMP_DIR / "batch_for_tts.txt"      # input for inference script
SPEAKER_ID   = os.getenv("EMOTIVOICE_SPK", "8051")  # any ID in speaker2id.txt
CHECKPOINT   = "g_00140000"                       # change if you like
LOGDIR       = "prompt_tts_open_source_joint"     # official default
CONFIG_DIR   = "config/joint"  # Use relative path for EmotiVoice script
EMO_NEUTRAL  = "Neutral"                          # baseline prompt

# ----------------------------- Load records ---------------------------------
jsonl_paths = sorted(JSONL_DIR.glob("*_results.jsonl"))
if not jsonl_paths:
    sys.exit(f"[ERROR] No *_results.jsonl under {JSONL_DIR}")

records: List[Dict] = []
for p in jsonl_paths:
    with p.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec["planning_model"] = p.stem
                records.append(rec)
print(f"Loaded {len(records)} planning records …")

# ------------------------- Build test‑file lines ----------------------------
lexicon = read_lexicon(f"{EMOTIVOICE_PATH}/lexicon/librispeech-lexicon.txt")
g2p = G2p()

lines: List[str] = []
index2meta: List[Tuple[str, str, str]] = []   # (pair_id, model_name, tag)

def to_phoneme(txt: str) -> str:
    """English g2p → token string understood by EmotiVoice."""
    return g2p_cn_en(txt, g2p, lexicon)       # returns e.g. "<sos/eos> [IH0] … <sos/eos>"

for rec in records:
    pid   = rec["pair_id"]
    model = rec["planning_model"]

    # baseline (neutral)
    text_b = rec["baseline_reply"].strip()
    ph_b   = to_phoneme(text_b)
    lines.append(f"{SPEAKER_ID}|{EMO_NEUTRAL}|{ph_b}|{text_b}")
    index2meta.append((pid, model, "neutral"))

    # emotion‑steered
    emo    = rec["model_emotion"].capitalize()
    text_s = rec["emotion_steered_reply"].strip()
    ph_s   = to_phoneme(text_s)
    lines.append(f"{SPEAKER_ID}|{emo}|{ph_s}|{text_s}")
    index2meta.append((pid, model, "steered"))

TEST_FILE.write_text("\n".join(lines), encoding="utf‑8")
print(f"[prep] Wrote {len(lines)} lines → {TEST_FILE}")

# ------------------------ Run official inference ---------------------------
# Create symbolic link to the models directory so the inference script can find it
outputs_dir = EMOTIVOICE_PATH / "outputs"
if not outputs_dir.exists():
    print(f"Creating symbolic link from {EMOTIVOICE_MODELS_PATH / 'outputs'} to {outputs_dir}")
    os.symlink(EMOTIVOICE_MODELS_PATH / "outputs", outputs_dir)

# Fix: Instead of using subprocess.run directly, create a shell script that properly sets up the environment
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

python inference_am_vocoder_joint.py \\
    --logdir {LOGDIR} \\
    --config_folder {CONFIG_DIR} \\
    --checkpoint {CHECKPOINT} \\
    --test_file {TEST_FILE}
"""
script_path.write_text(script_content)
os.chmod(script_path, 0o755)  # Make executable

print("[infer] Launching:", script_path)
subprocess.run(str(script_path), shell=True, check=True)
print("[infer] Inference finished.")

# ----------------------- Collect & rename WAVs ------------------------------
# Use the models path now since that's where the outputs will be
root_path  = EMOTIVOICE_MODELS_PATH / "outputs" / LOGDIR
wav_dir    = root_path / "test_audio" / "audio" / CHECKPOINT
wavs       = sorted(wav_dir.glob("*.wav"), key=lambda p: int(p.stem))

if len(wavs) != len(index2meta):
    sys.exit(f"[ERROR] Mismatch between WAV count ({len(wavs)}) and metadata list ({len(index2meta)}).")

meta_per_model: Dict[str, List[Dict]] = {}
for wav_path, (pid, model, tag) in zip(wavs, index2meta):
    tgt_dir  = OUT_ROOT / model
    tgt_dir.mkdir(parents=True, exist_ok=True)
    tgt_file = tgt_dir / f"{pid}_{tag}.wav"
    shutil.copy2(wav_path, tgt_file)

    meta_per_model.setdefault(model, []).append({
        "pair_id":        pid,
        "wav":            str(tgt_file),
        "emotion":        tag,
        "orig_path":      str(wav_path)
    })

# write metadata.jsonl like before
for model, entries in meta_per_model.items():
    meta_p = OUT_ROOT / model / "metadata.jsonl"
    with meta_p.open("w") as f:
        for m in entries:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[{model}] {len(entries)} entries → {meta_p}")

print("\nAll audio & metadata are under", OUT_ROOT)
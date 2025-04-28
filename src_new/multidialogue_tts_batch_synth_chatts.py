#!/usr/bin/env python3
"""
multidialogue_tts_batch_coqui.py  (Python 3.9 version)
------------------------------------------------------
• Reads every planning record in
      multidialog_emotion_planning/<model>/*results.jsonl
• For each record, creates two WAVs (neutral + steered)
  using Coqui TTS on your A6000.
• Writes 0-second silent WAVs whenever the text is blank,
  so downstream counts remain equal.
• Outputs live under multidialogue_coqui_out/<model>/…
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import soundfile as sf
from TTS.api import TTS
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama_3_3b_q4"

# Base directories
DATA_DIR  = Path("/data/group_data/starlight/gpa/tts")
JSONL_DIR = DATA_DIR / f"multidialog_emotion_planning/{DEFAULT_MODEL}"
OUT_ROOT  = DATA_DIR / "multidialogue_coqui_out"

# Coqui TTS detail
COQUI_MODEL = "tts_models/en/vctk/vits"

# ---------------------------------------------------------------------------


def write_silent_wav(path: Path, sample_rate: int = 22050) -> None:
    """Create a valid WAV file with zero-length data chunk (0-second audio)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(1, dtype=np.float32), sample_rate)
    with open(path, "r+b") as f:
        f.seek(44)          # skip RIFF header
        f.truncate()
    logger.info("✓ Wrote silent WAV → %s", path)


def generate_tts_audio(
    text: str,
    output_path: Path,
    tts: TTS,
    speaker: Optional[str] = None,
) -> None:
    """Synthesize text or write silent WAV when text is empty."""
    text = text.strip()
    if not text:
        write_silent_wav(output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if speaker:
        tts.tts_to_file(text=text, speaker=speaker, file_path=str(output_path))
    else:
        tts.tts_to_file(text=text, file_path=str(output_path))
    logger.info("✓ Saved audio → %s", output_path)


def main() -> None:
    # ── Load planning records ───────────────────────────────────────────────
    jsonl_files = sorted(JSONL_DIR.glob("*results.jsonl"))
    if not jsonl_files:
        logger.critical("No JSONL files found under %s", JSONL_DIR)
        sys.exit(1)

    records: List[Dict] = []
    for fn in jsonl_files:
        with fn.open() as f:
            for line in f:
                rec = json.loads(line)
                rec["planning_model"] = DEFAULT_MODEL
                records.append(rec)
    logger.info("Loaded %d planning records", len(records))

    # ── Init Coqui TTS ──────────────────────────────────────────────────────
    logger.info("Initializing Coqui TTS model %r on GPU", COQUI_MODEL)
    tts = TTS(model_name=COQUI_MODEL, progress_bar=False, gpu=True)

    speaker: Optional[str] = tts.speakers[0] if tts.is_multi_speaker else None
    if speaker:
        logger.info("Model is multi-speaker; using speaker %r", speaker)
    else:
        logger.info("Model is single-speaker; no speaker argument needed")

    # ── Synthesis loop ──────────────────────────────────────────────────────
    meta_per_model: Dict[str, List[Dict]] = {}
    total_jobs = len(records) * 2
    with tqdm(total=total_jobs, desc="Synthesizing TTS") as pbar:
        for rec in records:
            pid   = rec["pair_id"]
            model = rec["planning_model"]
            out_dir = OUT_ROOT / model

            # Neutral
            wav_n = out_dir / f"{pid}_neutral.wav"
            generate_tts_audio(rec["baseline_reply"], wav_n, tts, speaker)
            meta_per_model.setdefault(model, []).append({
                "pair_id":           pid,
                "wav":               str(wav_n),
                "emotion":           "neutral",
                "reference_emotion": rec.get("reference_emotion"),
            })
            pbar.update(1)

            # Steered
            wav_s = out_dir / f"{pid}_steered.wav"
            generate_tts_audio(rec["emotion_steered_reply"], wav_s, tts, speaker)
            meta_per_model[model].append({
                "pair_id":           pid,
                "wav":               str(wav_s),
                "emotion":           "steered",
                "reference_emotion": rec.get("reference_emotion"),
            })
            pbar.update(1)

    # ── Write metadata ──────────────────────────────────────────────────────
    for model, entries in meta_per_model.items():
        meta_path = OUT_ROOT / model / "metadata.jsonl"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        logger.info("Wrote metadata → %s", meta_path)

    logger.info("✅ Complete. Outputs in %s", OUT_ROOT)


if __name__ == "__main__":
    main()

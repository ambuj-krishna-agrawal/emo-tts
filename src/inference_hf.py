#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# inference_hf.py
# -----------------------------------------------------------------------------
# Synthesise speech for the 🤗  IEMOCAP dataset with a pre‑trained ESPnet model.
# -----------------------------------------------------------------------------
#  * Loads the dataset split with your Hugging Face token (env var or CLI).
#  * Writes WAV files to out_dir/<split>/<utt_id>.wav
#  * Uses a dataclass config for fast iteration.
# -----------------------------------------------------------------------------
from __future__ import annotations
import os
import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
import librosa
from tqdm import tqdm
from datasets import load_dataset
from espnet2.bin.tts_inference import Text2Speech

# Add NLTK data downloads for text processing
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# ───────────────────────── Global "quick‑edit" defaults ──────────────────────
DEFAULT_MODEL_TAG = "espnet/kan-bayashi_ljspeech_vits"  # Added espnet/ prefix
DEFAULT_SPLIT = "train"  # Only "train" is available for this dataset
DEFAULT_OUT_DIR = Path("generated_wavs")
DEFAULT_SR = 16_000  # force 16 kHz output


# ───────────────────────────── Config dataclass ──────────────────────────────
@dataclass
class Config:
    model_tag: str = DEFAULT_MODEL_TAG
    split: str = DEFAULT_SPLIT
    out_dir: Path = DEFAULT_OUT_DIR
    hf_token: Optional[str] = os.getenv("HF_TOKEN")  # env‑var fallback
    sample_rate: int = DEFAULT_SR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_samples: Optional[int] = None  # Limit number of samples to process


# ────────────────────────────────── Helpers ──────────────────────────────────
import json

def synthesise_dataset(cfg: Config):
    """Generate speech for every item in the chosen dataset split."""
    logging.info("Config:\n%s", cfg)

    # 1) Load ESPnet TTS model
    tts = Text2Speech.from_pretrained(
        model_tag=cfg.model_tag,
        device=cfg.device,
    )
    native_fs = tts.fs  # Get the model's native sample rate
    logging.info(f"Model's native sample rate: {native_fs} Hz")

    # 2) Load HF dataset
    ds = load_dataset("AbstractTTS/IEMOCAP",
                      split=cfg.split,
                      token=cfg.hf_token)

    # Inspect the dataset structure to find the correct field names
    example = ds[0]
    logging.info(f"Dataset example keys: {list(example.keys())}")
    logging.info(f"Dataset example values: {list(example.values())}")

    # If max_samples is set, limit the dataset size
    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))

    logging.info("Loaded %d examples from IEMOCAP %s split.",
                 len(ds), cfg.split)

    # 3) Output folders
    out_split_dir = cfg.out_dir / cfg.split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    
    # Create reference audio directory
    ref_split_dir = cfg.out_dir / "reference" / cfg.split
    ref_split_dir.mkdir(parents=True, exist_ok=True)

    # To store metadata for later evaluation
    metadata = []

    # 4) Synthesis loop - use appropriate field names based on inspection
    for i, ex in enumerate(tqdm(ds, desc="Synthesising")):
        # Extract the filename from the audio path to use as utterance_id
        file_path = ex.get("file")
        utt_id = Path(file_path).stem if file_path else f"utterance_{i:05d}"

        # Get text from appropriate field
        text = ex.get("transcription") or ex.get("text") or ex.get("sentence")
        if not text:
            logging.warning(f"No text found in example {i}. Skipping.")
            continue

        with torch.no_grad():
            result = tts(text)
            wav = result["wav"]

        wav_np = wav.view(-1).cpu().numpy()

        # Resample if needed
        if native_fs != cfg.sample_rate:
            wav_np = librosa.resample(wav_np, orig_sr=native_fs, target_sr=cfg.sample_rate)

        out_path = out_split_dir / f"{utt_id}.wav"
        sf.write(out_path, wav_np, cfg.sample_rate)

        # Save reference audio
        audio_data = ex.get("audio", {})
        ref_audio_array = audio_data.get("array", None)
        ref_audio_sr = audio_data.get("sampling_rate", cfg.sample_rate)
        
        ref_path = ref_split_dir / f"{utt_id}.wav"
        if ref_audio_array is not None:
            sf.write(ref_path, ref_audio_array, ref_audio_sr)
            logging.debug(f"Saved reference audio to {ref_path}")
        
        # Store comprehensive metadata including all emotion and acoustic values
        metadata_entry = {
            "utterance_id": utt_id,
            "transcript": text,
            "generated_wav_path": str(out_path),
            "reference_wav_path": str(ref_path),
            # Emotional labels
            "frustrated": ex.get("frustrated", 0),
            "angry": ex.get("angry", 0),
            "sad": ex.get("sad", 0),
            "disgust": ex.get("disgust", 0),
            "excited": ex.get("excited", 0),
            "fear": ex.get("fear", 0),
            "neutral": ex.get("neutral", 0),
            "surprise": ex.get("surprise", 0),
            "happy": ex.get("happy", 0),
            # Emotion dimensions
            "EmoAct": ex.get("EmoAct", 0),
            "EmoVal": ex.get("EmoVal", 0),
            "EmoDom": ex.get("EmoDom", 0),
            # Speaker info
            "gender": ex.get("gender", ""),
            "major_emotion": ex.get("major_emotion", ""),
            # Acoustic features
            "speaking_rate": ex.get("speaking_rate", 0),
            "pitch_mean": ex.get("pitch_mean", 0),
            "pitch_std": ex.get("pitch_std", 0),
            "rms": ex.get("rms", 0),
            "relative_db": ex.get("relative_db", 0)
        }
        
        metadata.append(metadata_entry)

    # Save metadata to a JSON file for later evaluation/comparison
    metadata_path = out_split_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Finished. WAVs and metadata saved to %s", out_split_dir)


# ─────────────────────────────────── Main ────────────────────────────────────
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="ESPnet inference on HF‑IEMOCAP")
    p.add_argument("--model_tag", default=DEFAULT_MODEL_TAG,
                   help="ESPnet model tag or local checkpoint")
    p.add_argument("--split", default=DEFAULT_SPLIT,
                   help="Dataset split to use")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN"),
                   help="HuggingFace auth token (env var HF_TOKEN is fallback)")
    p.add_argument("--sample_rate", type=int, default=DEFAULT_SR)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    p.add_argument("--max_samples", type=int, default=100,
                   help="Maximum number of samples to process")
    args = p.parse_args()

    return Config(**{k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = parse_args()
    synthesise_dataset(cfg)

# conda env export -n emo-tts-new --json > environment.json
# conda env export -n emo-tts-new --from-history > environment.yml

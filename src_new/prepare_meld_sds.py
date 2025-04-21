#!/usr/bin/env python3
"""
prepare_meld_sds.py

Prepare two-turn speaker pairs (A→B) from the MELD dataset for spoken-dialogue-system tasks.
Generates up to `max_pairs` adjacent-utterance pairs with speaker, emotion, transcript, and audio saved,
plus a metadata JSON for downstream ASR/TTS in ESPnet. Only pairs with different speakers and both
audio segments at least `min_duration` seconds are kept.
"""
import os
import argparse
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import load_dataset
import soundfile as sf
import librosa
from tqdm import tqdm

# ───────────────────────────── Global defaults ─────────────────────────────
DEFAULT_DATASET_NAME = "TwinkStart/MELD"
DEFAULT_SPLIT = "test"
DEFAULT_OUT_DIR = "meld_sds_pairs"
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_MAX_PAIRS = 100
DEFAULT_MIN_DURATION = 1.0  # seconds
DEFAULT_FILTER_DIFF_SPEAKER = True

# ───────────────────────────── Config dataclass ─────────────────────────────
@dataclass
class Config:
    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = DEFAULT_SPLIT
    out_dir: Path = Path(DEFAULT_OUT_DIR)
    hf_token: Optional[str] = DEFAULT_HF_TOKEN
    sample_rate: int = DEFAULT_SAMPLE_RATE
    max_pairs: int = DEFAULT_MAX_PAIRS
    min_duration: float = DEFAULT_MIN_DURATION
    require_different_speakers: bool = DEFAULT_FILTER_DIFF_SPEAKER

# ────────────────────────────────── Main logic ─────────────────────────────────
def prepare_pairs(cfg: Config):
    logging.info("Loading dataset %s split %s", cfg.dataset_name, cfg.split)
    ds = load_dataset(cfg.dataset_name, split=cfg.split, use_auth_token=cfg.hf_token)
    logging.info("Loaded %d examples", len(ds))

    # Group utterances by dialogue
    dialogues: dict[int, list] = {}
    for ex in ds:
        dlg_id = ex["Dialogue_ID"]
        dialogues.setdefault(dlg_id, []).append(ex)
    for dlg in dialogues.values():
        dlg.sort(key=lambda x: x["Utterance_ID"])

    # Build adjacent A→B pairs
    pairs = []
    for dlg_id, utts in dialogues.items():
        for idx in range(len(utts) - 1):
            pairs.append((utts[idx], utts[idx + 1]))
            if len(pairs) >= cfg.max_pairs:
                break
        if len(pairs) >= cfg.max_pairs:
            break
    logging.info("Generated %d raw pairs", len(pairs))

    # Apply filtering: different speakers and min duration
    filtered = []
    for a, b in pairs:
        if cfg.require_different_speakers and a["Speaker"] == b["Speaker"]:
            continue
        # extract audio arrays and sampling rates
        aud_a = a.get("audio", {}).get("array")
        sr_a = a.get("audio", {}).get("sampling_rate", cfg.sample_rate)
        aud_b = b.get("audio", {}).get("array")
        sr_b = b.get("audio", {}).get("sampling_rate", cfg.sample_rate)
        # check durations
        if aud_a is None or aud_b is None:
            continue
        dur_a = len(aud_a) / sr_a
        dur_b = len(aud_b) / sr_b
        if dur_a < cfg.min_duration or dur_b < cfg.min_duration:
            continue
        filtered.append((a, b))
        if len(filtered) >= cfg.max_pairs:
            break
    logging.info("Filtered down to %d pairs after speaker/duration constraints", len(filtered))

    # Prepare output directories
    audio_dir = cfg.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for i, (a, b) in enumerate(tqdm(filtered, desc="Saving pairs")):
        pair_id = f"{i:05d}"
        # Process utterance A
        aud_a = a["audio"]["array"]
        sr_a = a["audio"].get("sampling_rate", cfg.sample_rate)
        wav_a = aud_a
        if sr_a != cfg.sample_rate:
            wav_a = librosa.resample(wav_a, orig_sr=sr_a, target_sr=cfg.sample_rate)
        path_a = audio_dir / f"pair_{pair_id}_A.wav"
        sf.write(path_a, wav_a, cfg.sample_rate)

        # Process utterance B
        aud_b = b["audio"]["array"]
        sr_b = b["audio"].get("sampling_rate", cfg.sample_rate)
        wav_b = aud_b
        if sr_b != cfg.sample_rate:
            wav_b = librosa.resample(wav_b, orig_sr=sr_b, target_sr=cfg.sample_rate)
        path_b = audio_dir / f"pair_{pair_id}_B.wav"
        sf.write(path_b, wav_b, cfg.sample_rate)

        # Compile metadata
        entry = {
            "pair_id": pair_id,
            "dialogue_id": a["Dialogue_ID"],
            "utterance_id_A": a["Utterance_ID"],
            "utterance_id_B": b["Utterance_ID"],
            "speaker_A": a["Speaker"],
            "speaker_B": b["Speaker"],
            "emotion_A": a["Emotion"],
            "emotion_B": b["Emotion"],
            "sentiment_A": a.get("Sentiment"),
            "sentiment_B": b.get("Sentiment"),
            "text_A": a["Utterance"],
            "text_B": b["Utterance"],
            "duration_A": round(len(aud_a)/sr_a, 3),
            "duration_B": round(len(aud_b)/sr_b, 3),
            "audio_path_A": str(path_a),
            "audio_path_B": str(path_b),
            "video_path_A": a.get("video_path"),
            "video_path_B": b.get("video_path"),
        }
        metadata.append(entry)

    # Save metadata
    meta_path = cfg.out_dir / "pairs_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Saved metadata to %s", meta_path)

# ────────────────────────────────── Argument parsing ────────────────────────────
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Prepare SDS pairs from MELD dataset")
    parser.add_argument(
        "--dataset_name", default=DEFAULT_DATASET_NAME, required=False,
        help="HuggingFace dataset name (e.g., TwinkStart/MELD)"
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, required=False,
        help="Dataset split: train/dev/test"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path(DEFAULT_OUT_DIR), required=False,
        help="Output directory for pairs and metadata"
    )
    parser.add_argument(
        "--hf_token", default=DEFAULT_HF_TOKEN, required=False,
        help="HuggingFace auth token (env var HF_TOKEN fallback)"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, required=False,
        help="Target audio sample rate for saving wavs"
    )
    parser.add_argument(
        "--max_pairs", type=int, default=DEFAULT_MAX_PAIRS, required=False,
        help="Maximum number of A→B pairs to generate"
    )
    parser.add_argument(
        "--min_duration", type=float, default=DEFAULT_MIN_DURATION, required=False,
        help="Minimum duration (in seconds) for each utterance in a pair"
    )
    parser.add_argument(
        "--require_different_speakers", action="store_true", default=DEFAULT_FILTER_DIFF_SPEAKER,
        help="Only keep pairs where speaker A and B differ"
    )
    args = parser.parse_args()
    return Config(
        dataset_name=args.dataset_name,
        split=args.split,
        out_dir=args.out_dir,
        hf_token=args.hf_token,
        sample_rate=args.sample_rate,
        max_pairs=args.max_pairs,
        min_duration=args.min_duration,
        require_different_speakers=args.require_different_speakers,
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )
    cfg = parse_args()
    prepare_pairs(cfg)

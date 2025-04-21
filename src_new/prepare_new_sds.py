#!/usr/bin/env python3
"""
prepare_meld_audio_sds.py

Prepare two-turn speaker pairs (A→B) from the ajyy/MELD_audio dataset across train, validation, and test splits.
Generates up to `max_pairs` adjacent-utterance pairs with speaker, emotion, transcript, and audio saved,
plus a metadata JSON for downstream ASR/TTS in ESPnet. Only pairs with different speakers and both
audio segments at least `min_duration` seconds are kept.
"""
import os
import argparse
import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import pandas as pd
from datasets import load_dataset, DatasetDict
import soundfile as sf
import librosa
from tqdm import tqdm

# ───────────────────────────── Global defaults ─────────────────────────────
DEFAULT_DATASET_NAME = "ajyy/MELD_audio"
DEFAULT_CONFIG_NAME = "MELD_Audio"
DEFAULT_SPLITS = ["train","validation","test"]
DEFAULT_OUT_DIR = "meld_audio_sds_pairs_resampled"
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_PAIRS = 1000000000
DEFAULT_MIN_DURATION = 1.0
DEFAULT_FILTER_DIFF_SPEAKER = True
DEFAULT_METADATA_URL_TEMPLATE = (
    "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/{split}.csv"
)

# ───────────────────────────── Config dataclass ─────────────────────────────
@dataclass
class Config:
    dataset_name: str = DEFAULT_DATASET_NAME
    config_name: str = DEFAULT_CONFIG_NAME
    splits: List[str] = field(default_factory=lambda: DEFAULT_SPLITS.copy())
    out_dir: Path = Path(DEFAULT_OUT_DIR)
    hf_token: Optional[str] = DEFAULT_HF_TOKEN
    sample_rate: int = DEFAULT_SAMPLE_RATE
    max_pairs: int = DEFAULT_MAX_PAIRS
    min_duration: float = DEFAULT_MIN_DURATION
    require_different_speakers: bool = DEFAULT_FILTER_DIFF_SPEAKER
    metadata_url_template: str = DEFAULT_METADATA_URL_TEMPLATE

# ────────────────────────────────── Main logic ─────────────────────────────────
def prepare_pairs(cfg: Config):
    # 1) Load metadata CSVs for all splits
    meta_dict = {}
    for split in cfg.splits:
        # Map validation split name to dev.csv
        csv_name = "dev" if split == "validation" else split
        metadata_url = cfg.metadata_url_template.format(split=csv_name)
        logging.info("Loading metadata from %s", metadata_url)
        df = pd.read_csv(metadata_url, sep=",")
        logging.info("Loaded %d rows for split %s", len(df), split)
        for _, row in df.iterrows():
            did = int(row["Dialogue_ID"])
            uid = int(row["Utterance_ID"])
            key = f"dia{did}_utt{uid}"
            meta_dict[key] = {
                "dialogue_id": did,
                "utterance_id": uid,
                "speaker": row.get("Speaker"),
                "emotion": row.get("Emotion"),
                "sentiment": row.get("Sentiment"),
                "text": row.get("Utterance"),
            }
    logging.info("Combined metadata entries: %d", len(meta_dict))

    # 2) Load audio dataset
    logging.info("Loading dataset %s config %s", cfg.dataset_name, cfg.config_name)
    ds_dict: DatasetDict = load_dataset(
        cfg.dataset_name,
        cfg.config_name,
        use_auth_token=cfg.hf_token,
    )
    total = sum(len(ds_dict[s]) for s in ds_dict)
    logging.info("Loaded %d total examples across splits", total)

    # 3) Match dataset entries with metadata
    entries = []
    for split, ds in ds_dict.items():
        for ex in ds:
            path = ex.get("audio", {}).get("path")
            key = Path(path).stem if path else None
            if not key or key not in meta_dict:
                logging.warning("ID %s not in metadata, skipping", key)
                continue
            base = meta_dict[key].copy()
            base.update({
                "audio_array": ex["audio"]["array"],
                "audio_sr": ex["audio"]["sampling_rate"],
                "orig_audio_path": path,
            })
            entries.append(base)
    logging.info("Matched %d entries total", len(entries))

    # 4) Group by dialogue and sort
    dialogues = {}
    for item in entries:
        dialogues.setdefault(item["dialogue_id"], []).append(item)
    for utts in dialogues.values():
        utts.sort(key=lambda x: x["utterance_id"])

    # 5) Build raw pairs
    raw_pairs = []
    for utts in dialogues.values():
        for i in range(len(utts) - 1):
            raw_pairs.append((utts[i], utts[i + 1]))
            if len(raw_pairs) >= cfg.max_pairs:
                break
        if len(raw_pairs) >= cfg.max_pairs:
            break
    logging.info("Generated %d raw pairs", len(raw_pairs))

    # 6) Filter pairs by speaker and duration
    filtered = []
    for a, b in raw_pairs:
        if cfg.require_different_speakers and a["speaker"] == b["speaker"]:
            continue
        dur_a = len(a["audio_array"]) / a["audio_sr"]
        dur_b = len(b["audio_array"]) / b["audio_sr"]
        if dur_a < cfg.min_duration or dur_b < cfg.min_duration:
            continue
        filtered.append((a, b))
        if len(filtered) >= cfg.max_pairs:
            break
    logging.info("Filtered down to %d pairs", len(filtered))

    # 7) Save audio and metadata
    audio_dir = cfg.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for idx, (a, b) in enumerate(tqdm(filtered, desc="Saving pairs")):
        pid = f"{idx:05d}"
        # A
        wav_a = a["audio_array"]
        if a["audio_sr"] != cfg.sample_rate:
            wav_a = librosa.resample(
                wav_a, orig_sr=a["audio_sr"], target_sr=cfg.sample_rate
            )
        pa = audio_dir / f"pair_{pid}_A.wav"
        sf.write(pa, wav_a, cfg.sample_rate)
        # B
        wav_b = b["audio_array"]
        if b["audio_sr"] != cfg.sample_rate:
            wav_b = librosa.resample(
                wav_b, orig_sr=b["audio_sr"], target_sr=cfg.sample_rate
            )
        pb = audio_dir / f"pair_{pid}_B.wav"
        sf.write(pb, wav_b, cfg.sample_rate)

        metadata.append({
            "pair_id": pid,
            "dialogue_id": a["dialogue_id"],
            "utterance_id_A": a["utterance_id"],
            "utterance_id_B": b["utterance_id"],
            "speaker_A": a["speaker"],
            "speaker_B": b["speaker"],
            "emotion_A": a["emotion"],
            "emotion_B": b["emotion"],
            "sentiment_A": a.get("sentiment"),
            "sentiment_B": b.get("sentiment"),
            "text_A": a["text"],
            "text_B": b["text"],
            "duration_A": round(len(wav_a) / cfg.sample_rate, 3),
            "duration_B": round(len(wav_b) / cfg.sample_rate, 3),
            "audio_path_A": str(pa),
            "audio_path_B": str(pb),
            "orig_audio_path_A": a["orig_audio_path"],
            "orig_audio_path_B": b["orig_audio_path"],
        })
    meta_path = cfg.out_dir / "pairs_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Saved metadata to %s", meta_path)

# ────────────────────────────────── Argument parsing ────────────────────────────
def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Prepare SDS pairs from ajyy/MELD_audio across all splits"
    )
    parser.add_argument(
        "--dataset_name",
        default=DEFAULT_DATASET_NAME,
        required=False,
        help="Hugging Face dataset path",
    )
    parser.add_argument(
        "--config_name",
        default=DEFAULT_CONFIG_NAME,
        required=False,
        help="Builder config name for the dataset",
    )
    parser.add_argument(
        "--splits",
        default=','.join(DEFAULT_SPLITS),
        required=False,
        help="Comma-separated split names to combine",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(DEFAULT_OUT_DIR),
        required=False,
        help="Directory to save audio and metadata",
    )
    parser.add_argument(
        "--hf_token",
        default=DEFAULT_HF_TOKEN,
        required=False,
        help="Hugging Face auth token",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        required=False,
        help="Target WAV sample rate",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        required=False,
        help="Max number of pairs to generate",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        required=False,
        help="Min duration (seconds) per utterance",
    )
    parser.add_argument(
        "--require_different_speakers",
        action="store_true",
        default=DEFAULT_FILTER_DIFF_SPEAKER,
        help="Keep only pairs with different speakers",
    )
    parser.add_argument(
        "--metadata_url_template",
        default=DEFAULT_METADATA_URL_TEMPLATE,
        required=False,
        help="URL template for metadata CSV (use {split})",
    )
    args = parser.parse_args()
    return Config(
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        splits=[s.strip() for s in args.splits.split(',')],
        out_dir=args.out_dir,
        hf_token=args.hf_token,
        sample_rate=args.sample_rate,
        max_pairs=args.max_pairs,
        min_duration=args.min_duration,
        require_different_speakers=args.require_different_speakers,
        metadata_url_template=args.metadata_url_template,
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = parse_args()
    prepare_pairs(cfg)

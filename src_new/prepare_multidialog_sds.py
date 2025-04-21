#!/usr/bin/env python3
"""
prepare_multidialog_audio_sds.py

Prepare two-turn speaker pairs (A→B) from the IVLLab/MultiDialog dataset across all available splits.
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
from typing import Optional, List, Dict, Any, Tuple

from datasets import load_dataset, Dataset, DatasetDict
import soundfile as sf
import librosa
from tqdm import tqdm

# Configure more verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ───────────────────────────── Global defaults ─────────────────────────────
DEFAULT_DATASET_NAME = "IVLLab/MultiDialog"
DEFAULT_SPLITS = ["valid_freq", "valid_rare", "test_freq", "test_rare"]
DEFAULT_OUT_DIR = Path("/data/group_data/starlight/gpa/tts/multidialog_sds_pairs")
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_PAIRS = 1000000000
DEFAULT_MIN_DURATION = 1.0
DEFAULT_FILTER_DIFF_SPEAKER = True
DEFAULT_GOLD_EMOTION_ONLY = False
DEFAULT_GOLD_EMOTION_ACTORS = ["a", "b", "c", "e", "f", "g", "i", "j", "k"]

# ───────────────────────────── Config dataclass ─────────────────────────────
@dataclass
class Config:
    dataset_name: str = DEFAULT_DATASET_NAME
    splits: List[str] = field(default_factory=lambda: DEFAULT_SPLITS.copy())
    out_dir: Path = DEFAULT_OUT_DIR
    hf_token: Optional[str] = DEFAULT_HF_TOKEN
    sample_rate: int = DEFAULT_SAMPLE_RATE
    max_pairs: int = DEFAULT_MAX_PAIRS
    min_duration: float = DEFAULT_MIN_DURATION
    require_different_speakers: bool = DEFAULT_FILTER_DIFF_SPEAKER
    gold_emotion_only: bool = DEFAULT_GOLD_EMOTION_ONLY
    gold_emotion_actors: List[str] = field(default_factory=lambda: DEFAULT_GOLD_EMOTION_ACTORS.copy())

# ─────────────────── Helper functions ────────────────────
def get_speaker_id(file_name: str) -> str:
    """Extract speaker ID from file name (last character before .wav)"""
    try:
        # Debug
        logging.debug(f"Extracting speaker ID from filename: {file_name}")
        
        if not isinstance(file_name, str):
            # logging.warning(f"WARNING: file_name is not a string but {type(file_name)}")
            return "unknown"
            
        if file_name.endswith('.wav'):
            # Common pattern: last character before .wav
            speaker_id = file_name.split("_")[-1].split(".")[0][-1]
            # logging.debug(f"Extracted speaker ID: {speaker_id}")
            return speaker_id
        else:
            logging.warning(f"WARNING: file_name doesn't end with .wav: {file_name}")
            return "unknown"
    except Exception as e:
        logging.error(f"ERROR extracting speaker ID from {file_name}: {e}")
        return "unknown"

def group_by_conversation(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group examples by conversation ID and sort by utterance ID"""
    conversations = {}
    for ex in examples:
        conv_id = ex["conv_id"]
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(ex)
    
    # Sort each conversation by utterance ID
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x["utterance_id"])
    
    return conversations

def debug_print_example(ex, prefix="Example"):
    """Print the structure of an example for debugging"""
    logging.debug(f"\n{prefix}:")
    if isinstance(ex, dict):
        for k, v in ex.items():
            logging.debug(f"  {k}: {type(v)}")
            if k == "audio" and isinstance(v, dict):
                logging.debug(f"    Audio keys: {v.keys()}")
    else:
        logging.debug(f"  Type: {type(ex)}")
        logging.debug(f"  Value: {ex}")
    logging.debug("")
    
def debug_dataset_structure(ds):
    """Print the structure of a dataset for debugging"""
    logging.debug("\nDataset Structure:")
    if hasattr(ds, "features"):
        logging.debug(f"Features: {ds.features}")
    
    logging.debug(f"Dataset type: {type(ds)}")
    
    # Get an example if possible
    try:
        if len(ds) > 0:
            logging.debug("\nFirst example structure:")
            example = ds[0]
            debug_print_example(example)
    except Exception as e:
        logging.error(f"Error getting first example: {e}")
    
    logging.debug("")

# ────────────────────────────────── Load dataset ─────────────────────────────────
def load_dataset_split(cfg: Config, split: str) -> Dataset:
    cache_dir = "/data/group_data/starlight/gpa/tts/huggingface_cache"
    logging.info(f"=== Loading {split} split from HuggingFace Dataset ===")

    raw = load_dataset(
        cfg.dataset_name,
        split,            # this is being passed in as the config name
        cache_dir=cache_dir
    )

    # If HF returned a DatasetDict (with a single key == split), grab that subset
    if isinstance(raw, DatasetDict):
        ds = raw[split]
        logging.info(f"Extracted '{split}' subset with {len(ds)} examples")
    else:
        ds = raw

    return ds
# ────────────────────────────────── Main logic ─────────────────────────────────
def prepare_pairs(cfg: Config):
    # 1) Load dataset for all splits
    dataset_dict = {}
    
    for split in cfg.splits:
        logging.info(f"Loading dataset {cfg.dataset_name} split {split}")
        try:
            ds = load_dataset_split(cfg, split)
            dataset_dict[split] = ds
            logging.info(f"Loaded {len(ds)} examples for split {split}")
        except Exception as e:
            logging.error(f"Error loading split {split}: {e}")
    
    # 2) Convert dataset examples to our format
    all_examples = []
    for split, ds in dataset_dict.items():
        for ex in ds:
            try:
                # Get speaker ID from filename
                speaker_id = get_speaker_id(ex["file_name"])
                
                # Skip if not a gold emotion actor and we're filtering for those
                if cfg.gold_emotion_only and speaker_id not in cfg.gold_emotion_actors:
                    continue
                
                # Handle different audio formats in the dataset
                audio_array = None
                audio_sr = None
                
                if isinstance(ex["audio"], dict):
                    if "array" in ex["audio"] and "sampling_rate" in ex["audio"]:
                        # Standard HF datasets format
                        audio_array = ex["audio"]["array"]
                        audio_sr = ex["audio"]["sampling_rate"]
                    elif "path" in ex["audio"]:
                        # Path-only format
                        audio_path = ex["audio"]["path"]
                        try:
                            audio_array, audio_sr = librosa.load(audio_path, sr=None)
                        except Exception as e:
                            logging.error(f"Error loading audio file {audio_path}: {e}")
                            continue
                elif isinstance(ex["audio"], str):
                    # String path format
                    try:
                        audio_array, audio_sr = librosa.load(ex["audio"], sr=None)
                    except Exception as e:
                        logging.error(f"Error loading audio file {ex['audio']}: {e}")
                        continue
                
                # Skip if audio couldn't be loaded
                if audio_array is None or audio_sr is None:
                    logging.warning(f"Could not load audio for {ex['file_name']}")
                    continue
                    
                example = {
                    "split": split,
                    "conv_id": ex["conv_id"],
                    "utterance_id": ex["utterance_id"],
                    "speaker": ex["from"],  # 'human' or 'gpt'
                    "speaker_id": speaker_id,
                    "emotion": ex["emotion"],
                    "text": ex["value"],
                    "audio_array": audio_array,
                    "audio_sr": audio_sr,
                    "file_name": ex["file_name"],
                    "original_full_path": ex.get("original_full_path", "")
                }
                all_examples.append(example)
            except Exception as e:
                # Handle different error types
                if isinstance(ex, dict):
                    file_name = ex.get("file_name", "unknown")
                    logging.error(f"Error processing example {file_name}: {e}")
                elif isinstance(ex, str):
                    logging.error(f"Error processing example (string): {e}")
                else:
                    logging.error(f"Error processing example: {e}")
                continue
    
    logging.info(f"Collected {len(all_examples)} examples across all splits")
    
    # 3) Group by conversation and sort by utterance ID
    conversations = group_by_conversation(all_examples)
    logging.info(f"Grouped into {len(conversations)} conversations")
    
    # 4) Build raw pairs (adjacent utterances)
    raw_pairs = []
    for conv_id, utts in conversations.items():
        for i in range(len(utts) - 1):
            raw_pairs.append((utts[i], utts[i + 1]))
            if len(raw_pairs) >= cfg.max_pairs:
                break
        if len(raw_pairs) >= cfg.max_pairs:
            break
    logging.info(f"Generated {len(raw_pairs)} raw pairs")
    
    # 5) Filter pairs by speaker and duration
    filtered = []
    for a, b in raw_pairs:
        # Filter by different speakers if required
        if cfg.require_different_speakers and a["speaker"] == b["speaker"]:
            continue
            
        # Filter by minimum duration
        dur_a = len(a["audio_array"]) / a["audio_sr"]
        dur_b = len(b["audio_array"]) / b["audio_sr"]
        if dur_a < cfg.min_duration or dur_b < cfg.min_duration:
            continue
            
        filtered.append((a, b))
        if len(filtered) >= cfg.max_pairs:
            break
    
    logging.info(f"Filtered down to {len(filtered)} pairs")
    
    # 6) Save audio and metadata
    audio_dir = cfg.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    
    for idx, (a, b) in enumerate(tqdm(filtered, desc="Saving pairs")):
        pid = f"{idx:05d}"
        
        # Process and save first utterance audio
        wav_a = a["audio_array"]
        if a["audio_sr"] != cfg.sample_rate:
            wav_a = librosa.resample(
                wav_a, orig_sr=a["audio_sr"], target_sr=cfg.sample_rate
            )
        pa = audio_dir / f"pair_{pid}_A.wav"
        sf.write(pa, wav_a, cfg.sample_rate)
        
        # Process and save second utterance audio
        wav_b = b["audio_array"]
        if b["audio_sr"] != cfg.sample_rate:
            wav_b = librosa.resample(
                wav_b, orig_sr=b["audio_sr"], target_sr=cfg.sample_rate
            )
        pb = audio_dir / f"pair_{pid}_B.wav"
        sf.write(pb, wav_b, cfg.sample_rate)
        
        # Add metadata
        metadata.append({
            "pair_id": pid,
            "conv_id": a["conv_id"],
            "split": a["split"],
            "utterance_id_A": a["utterance_id"],
            "utterance_id_B": b["utterance_id"],
            "speaker_A": a["speaker"],
            "speaker_B": b["speaker"],
            "speaker_id_A": a["speaker_id"],
            "speaker_id_B": b["speaker_id"],
            "emotion_A": a["emotion"],
            "emotion_B": b["emotion"],
            "text_A": a["text"],
            "text_B": b["text"],
            "duration_A": round(len(wav_a) / cfg.sample_rate, 3),
            "duration_B": round(len(wav_b) / cfg.sample_rate, 3),
            "audio_path_A": str(pa),
            "audio_path_B": str(pb),
            "file_name_A": a["file_name"],
            "file_name_B": b["file_name"],
            "original_full_path_A": a["original_full_path"],
            "original_full_path_B": b["original_full_path"],
        })
    
    # Save metadata to JSON
    meta_path = cfg.out_dir / "pairs_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata to {meta_path}")
    
    # Generate a summary
    logging.info(f"Summary:")
    logging.info(f"  Total pairs: {len(metadata)}")
    logging.info(f"  Unique conversations: {len(set(m['conv_id'] for m in metadata))}")
    
    # Count pairs by split
    split_counts = {}
    for m in metadata:
        split = m["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
    for split, count in split_counts.items():
        logging.info(f"  Pairs from {split}: {count}")
    
    # Count emotion distribution
    emotion_counts_a = {}
    emotion_counts_b = {}
    for m in metadata:
        emotion_a = m["emotion_A"]
        emotion_b = m["emotion_B"]
        emotion_counts_a[emotion_a] = emotion_counts_a.get(emotion_a, 0) + 1
        emotion_counts_b[emotion_b] = emotion_counts_b.get(emotion_b, 0) + 1
    
    logging.info(f"  Emotion distribution for first utterance:")
    for emotion, count in sorted(emotion_counts_a.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"    {emotion}: {count} ({count/len(metadata)*100:.1f}%)")
    
    logging.info(f"  Emotion distribution for second utterance:")
    for emotion, count in sorted(emotion_counts_b.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"    {emotion}: {count} ({count/len(metadata)*100:.1f}%)")

# ────────────────────────────────── Argument parsing ────────────────────────────
def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Prepare SDS pairs from IVLLab/MultiDialog across all splits"
    )
    parser.add_argument(
        "--dataset_name",
        default=DEFAULT_DATASET_NAME,
        required=False,
        help="Hugging Face dataset path",
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
        "--gold_emotion_only",
        action="store_true",
        default=DEFAULT_GOLD_EMOTION_ONLY,
        help="Use only gold emotion actors (>40% emotion accuracy)",
    )
    parser.add_argument(
        "--gold_emotion_actors",
        default=','.join(DEFAULT_GOLD_EMOTION_ACTORS),
        required=False,
        help="Comma-separated list of gold emotion actor IDs",
    )
    
    args = parser.parse_args()
    return Config(
        dataset_name=args.dataset_name,
        splits=[s.strip() for s in args.splits.split(',')],
        out_dir=args.out_dir,
        hf_token=args.hf_token,
        sample_rate=args.sample_rate,
        max_pairs=args.max_pairs,
        min_duration=args.min_duration,
        require_different_speakers=args.require_different_speakers,
        gold_emotion_only=args.gold_emotion_only,
        gold_emotion_actors=[s.strip() for s in args.gold_emotion_actors.split(',')],
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = parse_args()
    prepare_pairs(cfg)
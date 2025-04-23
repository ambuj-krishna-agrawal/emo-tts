#!/usr/bin/env python3
"""
prepare_multidialog_audio_sds.py

Prepare two-turn speaker pairs (A→B) from the IVLLab/MultiDialog dataset across all available splits.
Generates up to `max_pairs` adjacent-utterance pairs with speaker, emotion, transcript, and audio saved,
plus a metadata JSON for downstream ASR/TTS in ESPnet. Only pairs with different speakers and both
audio segments at least `min_duration` seconds are kept.

Pairs are filtered based on semantic relevance to identify high-quality pairs
where speaker B shows responses to speaker A's utterances.
"""
import os
import argparse
import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from datasets import load_dataset, Dataset, DatasetDict
import soundfile as sf
import librosa
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import string

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ───────────────────────────── Global defaults ─────────────────────────────
DEFAULT_DATASET_NAME = "IVLLab/MultiDialog"
DEFAULT_SPLITS = ["valid_freq", "valid_rare", "test_freq", "test_rare"]
DEFAULT_OUT_DIR = Path("/data/group_data/starlight/gpa/tts/multidialog_sds_pairs")
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_DURATION = 1.0
DEFAULT_FILTER_DIFF_SPEAKER = True
DEFAULT_GOLD_EMOTION_ONLY = False
DEFAULT_GOLD_EMOTION_ACTORS = ["a", "b", "c", "e", "f", "g", "i", "j", "k"]
DEFAULT_TOP_N_PAIRS = 1000
DEFAULT_MIN_WORD_COUNT = 15
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.2  # Minimum semantic similarity between A and B
DEFAULT_TARGET_EMOTIONS = [
    "Anger", "Fear", "Surprise", "Disgust", "Sadness", "Joy"
]  # Emotions to prioritize
DEFAULT_NEUTRAL_RATIO = 0.05  # Reduced neutral ratio

# ───────────────────────────── Config dataclass ─────────────────────────────
@dataclass
class Config:
    dataset_name: str = DEFAULT_DATASET_NAME
    splits: List[str] = field(default_factory=lambda: DEFAULT_SPLITS.copy())
    out_dir: Path = DEFAULT_OUT_DIR
    hf_token: Optional[str] = DEFAULT_HF_TOKEN
    sample_rate: int = DEFAULT_SAMPLE_RATE
    min_duration: float = DEFAULT_MIN_DURATION
    require_different_speakers: bool = DEFAULT_FILTER_DIFF_SPEAKER
    gold_emotion_only: bool = DEFAULT_GOLD_EMOTION_ONLY
    gold_emotion_actors: List[str] = field(default_factory=lambda: DEFAULT_GOLD_EMOTION_ACTORS.copy())
    top_n_pairs: int = DEFAULT_TOP_N_PAIRS
    min_word_count: int = DEFAULT_MIN_WORD_COUNT
    semantic_similarity_threshold: float = DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD
    target_emotions: List[str] = field(default_factory=lambda: DEFAULT_TARGET_EMOTIONS.copy())
    neutral_ratio: float = DEFAULT_NEUTRAL_RATIO

# ─────────────────── Helper functions ────────────────────
def get_speaker_id(file_name: str) -> str:
    """Extract speaker ID from file name (last character before .wav)"""
    try:
        if not isinstance(file_name, str):
            return "unknown"
            
        if file_name.endswith('.wav'):
            # Common pattern: last character before .wav
            speaker_id = file_name.split("_")[-1].split(".")[0][-1]
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

def count_words(text: str) -> int:
    """Count the number of words in a string"""
    return len(text.split())

def calculate_semantic_similarity(text_a: str, text_b: str) -> float:
    """Calculate semantic similarity between two texts using TF-IDF and cosine similarity"""
    if not text_a or not text_b:
        return 0.0
        
    # Clean and normalize texts
    def clean_text(text):
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
        
    text_a_clean = clean_text(text_a)
    text_b_clean = clean_text(text_b)
    
    # If after cleaning texts are too short, similarity is low
    if len(text_a_clean.split()) < 3 or len(text_b_clean.split()) < 3:
        return 0.1
    
    # Create TF-IDF vectorizer and calculate similarity
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text_a_clean, text_b_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        # If vectorization fails, return 0
        return 0.0

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
    # 1) Load dataset for all splits -------------------------------------------------
    dataset_dict = {}
    for split in cfg.splits:
        logging.info(f"Loading dataset {cfg.dataset_name} split {split}")
        try:
            ds = load_dataset_split(cfg, split)
            dataset_dict[split] = ds
            logging.info(f"Loaded {len(ds)} examples for split {split}")
        except Exception as e:
            logging.error(f"Error loading split {split}: {e}")

    # 2) Convert dataset examples to our format -------------------------------------
    all_examples = []
    for split, ds in dataset_dict.items():
        for ex in ds:
            try:
                speaker_id = get_speaker_id(ex["file_name"])

                if cfg.gold_emotion_only and speaker_id not in cfg.gold_emotion_actors:
                    continue

                # --- load / resolve audio ------------------------------------------
                audio_array, audio_sr = None, None
                if isinstance(ex["audio"], dict):
                    if {"array", "sampling_rate"} <= ex["audio"].keys():
                        audio_array = ex["audio"]["array"]
                        audio_sr    = ex["audio"]["sampling_rate"]
                    elif "path" in ex["audio"]:
                        audio_array, audio_sr = librosa.load(ex["audio"]["path"], sr=None)
                elif isinstance(ex["audio"], str):
                    audio_array, audio_sr = librosa.load(ex["audio"], sr=None)

                if audio_array is None or audio_sr is None:
                    logging.warning(f"Could not load audio for {ex['file_name']}")
                    continue
                # -------------------------------------------------------------------

                all_examples.append(
                    {
                        "split":          split,
                        "conv_id":        ex["conv_id"],
                        "utterance_id":   ex["utterance_id"],
                        "speaker":        ex["from"],
                        "speaker_id":     speaker_id,
                        "emotion":        ex["emotion"],
                        "text":           ex["value"],
                        "audio_array":    audio_array,
                        "audio_sr":       audio_sr,
                        "file_name":      ex["file_name"],
                        "original_full_path": ex.get("original_full_path", ""),
                    }
                )
            except Exception as e:
                logging.error(f"Error processing example {ex.get('file_name','?')}: {e}")
                continue

    logging.info(f"Collected {len(all_examples)} examples across all splits")

    # 3) Group by conversation -------------------------------------------------------
    conversations = group_by_conversation(all_examples)
    logging.info(f"Grouped into {len(conversations)} conversations")

    # 4) Build raw adjacent‑utterance pairs -----------------------------------------
    raw_pairs = [
        (utts[i], utts[i + 1])
        for utts in conversations.values()
        for i in range(len(utts) - 1)
    ]
    logging.info(f"Generated {len(raw_pairs)} raw pairs")

    # 5) Basic filtering: speakers, duration, word‑count, **neutral B** -------------
    basic_filtered_pairs = []
    for a, b in raw_pairs:
        # if cfg.require_different_speakers and a["speaker"] == b["speaker"]:
        #     continue

        # Skip pairs where B's emotion is Neutral
        if b["emotion"] == "Neutral":
            continue

        dur_a = len(a["audio_array"]) / a["audio_sr"]
        dur_b = len(b["audio_array"]) / b["audio_sr"]
        if dur_a < cfg.min_duration or dur_b < cfg.min_duration:
            continue

        if count_words(a["text"]) < cfg.min_word_count or count_words(b["text"]) < cfg.min_word_count:
            continue

        basic_filtered_pairs.append((a, b))

    logging.info(f"Filtered down to {len(basic_filtered_pairs)} pairs by speaker, duration, word‑count, and non‑neutral B")

    # 6) Calculate semantic similarity only
    scored_pairs = []
    for a, b in basic_filtered_pairs:
        semantic_similarity = calculate_semantic_similarity(a["text"], b["text"])
        scored_pairs.append(
            (
                a,
                b,
                {
                    "semantic_similarity": semantic_similarity,
                    "is_target_emotion": b["emotion"] in cfg.target_emotions,
                },
            )
        )
    logging.info(f"Calculated semantic similarity for {len(scored_pairs)} pairs")

    # 7) Apply semantic similarity threshold filtering
    similarity_filtered = [
        (a, b, s) for a, b, s in scored_pairs if s["semantic_similarity"] >= cfg.semantic_similarity_threshold
    ]
    logging.info(f"{len(similarity_filtered)} pairs ≥ similarity {cfg.semantic_similarity_threshold}")

    # 8) Keep target emotions only
    target_emotion_pairs = [
        (a, b, s) for a, b, s in similarity_filtered if s["is_target_emotion"]
    ]
    logging.info(f"{len(target_emotion_pairs)} pairs with target emotions")

    # If we have no pairs with target emotions, use all pairs that passed similarity filter
    if len(target_emotion_pairs) == 0:
        logging.warning("No pairs with target emotions found, using all similarity-filtered pairs")
        target_emotion_pairs = similarity_filtered

    # 9) Rank by similarity, diversify by emotion, select top-N
    # Sort by semantic similarity in descending order
    target_emotion_pairs.sort(key=lambda x: x[2]["semantic_similarity"], reverse=True)

    # Bucket pairs by emotion
    emotion_buckets: Dict[str, List[Tuple[dict, dict, dict]]] = {}
    for p in target_emotion_pairs:
        emotion_buckets.setdefault(p[1]["emotion"], []).append(p)

    # Select pairs from each emotion bucket
    final_pairs: List[Tuple[dict, dict]] = []
    if emotion_buckets:
        per_emotion = max(2, cfg.top_n_pairs // len(emotion_buckets))
        for emo, bucket in emotion_buckets.items():
            final_pairs.extend([(a, b) for a, b, _ in bucket[:per_emotion]])
            
        # If we don't have enough pairs, add more from the sorted list
        if len(final_pairs) < cfg.top_n_pairs:
            seen = {(a["conv_id"], a["utterance_id"], b["utterance_id"]) for a, b in final_pairs}
            for a, b, _ in target_emotion_pairs:
                key = (a["conv_id"], a["utterance_id"], b["utterance_id"])
                if key not in seen and len(final_pairs) < cfg.top_n_pairs:
                    final_pairs.append((a, b))
                    seen.add(key)

    # 10) Shuffle, log distribution, save audio/metadata (unchanged) ---------------
    np.random.shuffle(final_pairs)
    emotion_counts = {}
    for _, b in final_pairs:
        emotion_counts[b["emotion"]] = emotion_counts.get(b["emotion"], 0) + 1
    logging.info("Final emotion distribution:")
    for emo, cnt in emotion_counts.items():
        logging.info(f"  {emo}: {cnt}  ({cnt/len(final_pairs):.1%})")

    audio_dir = cfg.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for idx, (a, b) in enumerate(tqdm(final_pairs, desc="Saving pairs")):
        pid = f"{idx:05d}"
        pa = audio_dir / f"pair_{pid}_A.wav"
        pb = audio_dir / f"pair_{pid}_B.wav"

        wav_a = a["audio_array"]
        if a["audio_sr"] != cfg.sample_rate:
            wav_a = librosa.resample(wav_a, orig_sr=a["audio_sr"], target_sr=cfg.sample_rate)
        sf.write(pa, wav_a, cfg.sample_rate)

        wav_b = b["audio_array"]
        if b["audio_sr"] != cfg.sample_rate:
            wav_b = librosa.resample(wav_b, orig_sr=b["audio_sr"], target_sr=cfg.sample_rate)
        sf.write(pb, wav_b, cfg.sample_rate)

        metadata.append(
            {
                "pair_id":          pid,
                "conv_id":          a["conv_id"],
                "split":            a["split"],
                "utterance_id_A":   a["utterance_id"],
                "utterance_id_B":   b["utterance_id"],
                "speaker_A":        a["speaker"],
                "speaker_B":        b["speaker"],
                "speaker_id_A":     a["speaker_id"],
                "speaker_id_B":     b["speaker_id"],
                "emotion_A":        a["emotion"],
                "emotion_B":        b["emotion"],
                "text_A":           a["text"],
                "text_B":           b["text"],
                "word_count_A":     count_words(a["text"]),
                "word_count_B":     count_words(b["text"]),
                "duration_A":       round(len(wav_a) / cfg.sample_rate, 3),
                "duration_B":       round(len(wav_b) / cfg.sample_rate, 3),
                "audio_path_A":     str(pa),
                "audio_path_B":     str(pb),
                "file_name_A":      a["file_name"],
                "file_name_B":      b["file_name"],
                "original_full_path_A": a["original_full_path"],
                "original_full_path_B": b["original_full_path"],
            }
        )

    meta_path = cfg.out_dir / "pairs_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata to {meta_path}")

# ────────────────────────────────── Argument parsing ────────────────────────────
def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Prepare SDS pairs from IVLLab/MultiDialog with semantic relevance filtering"
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
    parser.add_argument(
        "--top_n_pairs",
        type=int,
        default=DEFAULT_TOP_N_PAIRS,
        required=False,
        help="Keep only the top N pairs with the strongest emotions",
    )
    parser.add_argument(
        "--min_word_count",
        type=int,
        default=DEFAULT_MIN_WORD_COUNT,
        required=False,
        help="Minimum number of words per utterance",
    )
    parser.add_argument(
        "--semantic_similarity_threshold",
        type=float,
        default=DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
        required=False,
        help="Minimum semantic similarity between speaker A and B utterances",
    )
    parser.add_argument(
        "--target_emotions",
        default=','.join(DEFAULT_TARGET_EMOTIONS),
        required=False,
        help="Comma-separated list of target emotions to prioritize",
    )
    parser.add_argument(
        "--neutral_ratio",
        type=float,
        default=DEFAULT_NEUTRAL_RATIO,
        required=False,
        help="Maximum ratio of neutral emotion pairs to include",
    )
    
    args = parser.parse_args()
    
    return Config(
        dataset_name=args.dataset_name,
        splits=[s.strip() for s in args.splits.split(',')],
        out_dir=args.out_dir,
        hf_token=args.hf_token,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        require_different_speakers=args.require_different_speakers,
        gold_emotion_only=args.gold_emotion_only,
        gold_emotion_actors=[s.strip() for s in args.gold_emotion_actors.split(',')],
        top_n_pairs=args.top_n_pairs,
        min_word_count=args.min_word_count,
        semantic_similarity_threshold=args.semantic_similarity_threshold,
        target_emotions=[s.strip() for s in args.target_emotions.split(',')],
        neutral_ratio=args.neutral_ratio,
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = parse_args()
    prepare_pairs(cfg)
#!/usr/bin/env python3
"""
prepare_multidialog_audio_sds.py

Prepare two-turn speaker pairs (A→B) from the IVLLab/MultiDialog dataset across all available splits.
Generates up to `max_pairs` adjacent-utterance pairs with speaker, emotion, transcript, and audio saved,
plus a metadata JSON for downstream ASR/TTS in ESPnet. Only pairs with different speakers and both
audio segments at least `min_duration` seconds are kept.

Pairs are filtered based on semantic relevance and emotion strength to identify high-quality pairs
where speaker B shows strong emotional responses to speaker A's utterances.
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
DEFAULT_TOP_N_PAIRS = 20
DEFAULT_MIN_WORD_COUNT = 15
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.2  # Minimum semantic similarity between A and B
DEFAULT_EMOTION_STRENGTH_THRESHOLD = 0.5      # Increased minimum emotion strength score
DEFAULT_TARGET_EMOTIONS = [
    "Anger", "Fear", "Surprise", "Disgust", "Sadness", "Joy"
]  # Emotions to prioritize
DEFAULT_NEUTRAL_RATIO = 0.1  # Reduced neutral ratio
DEFAULT_MIN_A_EMOTION_STRENGTH = 0.3  # Minimum emotion strength for speaker A

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
    emotion_strength_threshold: float = DEFAULT_EMOTION_STRENGTH_THRESHOLD
    min_a_emotion_strength: float = DEFAULT_MIN_A_EMOTION_STRENGTH
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

def score_emotion_strength(text: str, emotion: str) -> float:
    """
    Score the emotional strength of a text based on sentiment analysis
    and the stated emotion label - improved to better detect strong emotions
    """
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    
    # Map emotions to expected sentiment patterns
    emotional_intensity = 0.0
    
    # Use compound score for overall intensity - higher initial weight
    base_intensity = abs(sentiment_scores['compound']) * 1.2
    
    # Boost intensity based on matching emotion label with sentiment
    if emotion in ["Anger", "Disgust"] and sentiment_scores['neg'] > 0.2:
        emotional_intensity = base_intensity * 2.0
    elif emotion == "Joy" and sentiment_scores['pos'] > 0.2:
        emotional_intensity = base_intensity * 2.0
    elif emotion == "Sadness" and sentiment_scores['neg'] > 0.2:
        emotional_intensity = base_intensity * 1.8
    elif emotion == "Surprise" and (sentiment_scores['pos'] > 0.2 or sentiment_scores['neg'] > 0.2):
        emotional_intensity = base_intensity * 1.8
    elif emotion == "Fear" and sentiment_scores['neg'] > 0.2:
        emotional_intensity = base_intensity * 2.0
    elif emotion == "Neutral":
        # Significantly lower score for neutral emotions
        emotional_intensity = base_intensity * 0.3
    else:
        # Default case - still boosted slightly
        emotional_intensity = base_intensity * 1.2
    
    # Add stronger bonus for emotional words and punctuation
    emotional_markers = {
        '!': 0.15,                   # Exclamation
        '?!': 0.2,                   # Surprised question
        '!!!': 0.25,                 # Multiple exclamations
        'very': 0.15,                # Intensity words
        'really': 0.15,
        'extremely': 0.2,
        'absolutely': 0.2,
        'definitely': 0.15,
        'furious': 0.25,             # Strong emotion words
        'terrified': 0.25,
        'ecstatic': 0.25,
        'devastated': 0.25,
        'thrilled': 0.2,
        'hate': 0.2,
        'love': 0.2,
        'horrible': 0.2,
        'amazing': 0.2,
        'terrible': 0.2,
        'awful': 0.2,
        'incredible': 0.2,
        'worst': 0.2,
        'best': 0.2,
    }
    
    for marker, bonus in emotional_markers.items():
        if marker in text.lower():
            emotional_intensity += bonus
    
    # Check for ALL CAPS words (indicating emphasis/shouting)
    words = text.split()
    for word in words:
        if len(word) > 2 and word.isupper():
            emotional_intensity += 0.25
            break
    
    # Check for repeated punctuation
    if '!!' in text or '??' in text:
        emotional_intensity += 0.2
    
    # Longer texts get a slight bonus if they're already somewhat emotional
    if len(text) > 50 and emotional_intensity > 0.4:
        emotional_intensity += 0.1
    
    # Cap at 1.0
    return min(emotional_intensity, 1.0)

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
    logging.info(f"Generated {len(raw_pairs)} raw pairs")
    
    # 5) Filter pairs by speaker, duration, and minimum word count
    basic_filtered_pairs = []
    for a, b in raw_pairs:
        # Filter by different speakers if required
        if cfg.require_different_speakers and a["speaker"] == b["speaker"]:
            continue
            
        # Filter by minimum duration
        dur_a = len(a["audio_array"]) / a["audio_sr"]
        dur_b = len(b["audio_array"]) / b["audio_sr"]
        if dur_a < cfg.min_duration or dur_b < cfg.min_duration:
            continue
        
        # Filter by minimum word count
        word_count_a = count_words(a["text"])
        word_count_b = count_words(b["text"])
        if word_count_a < cfg.min_word_count or word_count_b < cfg.min_word_count:
            continue
            
        basic_filtered_pairs.append((a, b))
    
    logging.info(f"Filtered down to {len(basic_filtered_pairs)} pairs by speaker, duration, and minimum word count requirements")
    
    # 6) Calculate semantic similarity and emotion strength scores for each pair
    scored_pairs = []
    for a, b in basic_filtered_pairs:
        # Calculate semantic similarity between A and B
        semantic_similarity = calculate_semantic_similarity(a["text"], b["text"])
        
        # Calculate emotion strength score for speaker B
        emotion_strength = score_emotion_strength(b["text"], b["emotion"])
        
        # Create score dictionary
        score_dict = {
            "semantic_similarity": semantic_similarity,
            "emotion_strength": emotion_strength,
            "is_target_emotion": b["emotion"] in cfg.target_emotions,
            "is_neutral": b["emotion"] == "Neutral"
        }
        
        # Add to scored pairs
        scored_pairs.append((a, b, score_dict))
    
    logging.info(f"Calculated semantic similarity and emotion strength for {len(scored_pairs)} pairs")
    
    # 7) Filter by semantic similarity threshold
    similarity_filtered = [
        (a, b, scores) for a, b, scores in scored_pairs 
        if scores["semantic_similarity"] >= cfg.semantic_similarity_threshold
    ]
    logging.info(f"Filtered to {len(similarity_filtered)} pairs with semantic similarity >= {cfg.semantic_similarity_threshold}")
    
    # 8) Filter for emotion strength in speaker A and ensure non-neutral for speaker B
    strong_emotion_pairs = []
    
    for a, b, scores in similarity_filtered:
        # Calculate emotion strength for speaker A
        a_emotion_strength = score_emotion_strength(a["text"], a["emotion"])
        
        # Skip pairs where speaker B has neutral emotion (critical change)
        if b["emotion"] == "Neutral":
            continue
            
        # Skip pairs where speaker A's emotion is too weak
        if a_emotion_strength < cfg.min_a_emotion_strength:
            continue
        
        # Keep only pairs with strong enough speaker B emotion
        if scores["is_target_emotion"] and scores["emotion_strength"] >= cfg.emotion_strength_threshold:
            # Add speaker A emotion strength to the scores
            scores["a_emotion_strength"] = a_emotion_strength
            strong_emotion_pairs.append((a, b, scores))
    
    # Keep a very small number of neutral pairs if needed for comparison
    neutral_emotion_pairs = []
    if cfg.neutral_ratio > 0:
        for a, b, scores in similarity_filtered:
            if b["emotion"] == "Neutral" and scores["semantic_similarity"] > cfg.semantic_similarity_threshold * 1.5:
                neutral_emotion_pairs.append((a, b, scores))
    
    logging.info(f"Selected {len(strong_emotion_pairs)} strong emotion pairs and {len(neutral_emotion_pairs)} neutral emotion pairs for comparison")
    
    # 9) Sort by combined emotion strength (speaker A + speaker B) for better overall emotional content
    sorted_emotion_pairs = sorted(
        strong_emotion_pairs, 
        key=lambda x: x[2]["emotion_strength"] + x[2]["a_emotion_strength"], 
        reverse=True
    )
    
    # Further group by emotion type to ensure diversity
    emotion_grouped_pairs = {}
    for pair in sorted_emotion_pairs:
        emotion = pair[1]["emotion"]
        if emotion not in emotion_grouped_pairs:
            emotion_grouped_pairs[emotion] = []
        emotion_grouped_pairs[emotion].append(pair)
    
    # Sort neutrals by semantic similarity (descending) if we're keeping any
    sorted_neutral_pairs = sorted(
        neutral_emotion_pairs,
        key=lambda x: x[2]["semantic_similarity"],
        reverse=True
    )
    
    # 10) Select top pairs from each emotion to ensure diversity
    diverse_selection = []
    
    # Calculate how many pairs to select per emotion
    emotions_present = len(emotion_grouped_pairs)
    if emotions_present > 0:
        pairs_per_emotion = max(2, cfg.top_n_pairs // emotions_present)  # At least 2 per emotion
        
        # Take top pairs from each emotion
        for emotion, pairs in emotion_grouped_pairs.items():
            diverse_selection.extend(pairs[:pairs_per_emotion])
            
        # If we haven't selected enough, add more from the overall sorted list
        if len(diverse_selection) < cfg.top_n_pairs:
            # Create a set of already selected pairs
            selected_indices = set((p[0]["conv_id"], p[0]["utterance_id"], p[1]["utterance_id"]) 
                                for p in diverse_selection)
            
            # Add more pairs that aren't already selected
            for pair in sorted_emotion_pairs:
                pair_key = (pair[0]["conv_id"], pair[0]["utterance_id"], pair[1]["utterance_id"])
                if pair_key not in selected_indices and len(diverse_selection) < cfg.top_n_pairs:
                    diverse_selection.append(pair)
                    selected_indices.add(pair_key)
    
    # Calculate how many pairs of each type to include
    strong_count = min(len(diverse_selection), int(cfg.top_n_pairs * (1 - cfg.neutral_ratio)))
    neutral_count = min(len(sorted_neutral_pairs), cfg.top_n_pairs - strong_count)
    
    logging.info(f"Selecting top {strong_count} strong emotion pairs with diversity and top {neutral_count} neutral comparison pairs")
    
    # 11) Combine the selected pairs
    final_pairs = [(a, b) for a, b, _ in diverse_selection[:strong_count]]
    
    # Only add neutral pairs if we want them for comparison
    if neutral_count > 0:
        final_pairs.extend([(a, b) for a, b, _ in sorted_neutral_pairs[:neutral_count]])
    
    # 12) Shuffle the final pairs to mix emotions
    np.random.shuffle(final_pairs)
    
    # 13) Add debug info about the final pairs
    if final_pairs:
        logging.info("Example selected pairs:")
        for i, (a, b) in enumerate(final_pairs[:3]):  # Show first 3 examples
            logging.info(f"Example {i+1}:")
            logging.info(f"  Speaker A ({a['emotion']}): {a['text'][:100]}...")
            logging.info(f"  Speaker B ({b['emotion']}): {b['text'][:100]}...")
    
    # 13) Log emotion distribution in final pairs
    emotion_counts = {}
    for a, b in final_pairs:
        emotion = b["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    logging.info("Final emotion distribution:")
    for emotion, count in emotion_counts.items():
        logging.info(f"  {emotion}: {count} ({count/len(final_pairs)*100:.1f}%)")
    
    # 14) Save audio and metadata
    audio_dir = cfg.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    
    for idx, (a, b) in enumerate(tqdm(final_pairs, desc="Saving pairs")):
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
            "word_count_A": count_words(a["text"]),
            "word_count_B": count_words(b["text"]),
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

# ────────────────────────────────── Argument parsing ────────────────────────────
def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Prepare SDS pairs from IVLLab/MultiDialog with semantic relevance and emotion filtering"
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
        "--emotion_strength_threshold",
        type=float,
        default=DEFAULT_EMOTION_STRENGTH_THRESHOLD,
        required=False,
        help="Minimum emotion strength score for target emotions in speaker B",
    )
    parser.add_argument(
        "--min_a_emotion_strength",
        type=float,
        default=DEFAULT_MIN_A_EMOTION_STRENGTH,
        required=False,
        help="Minimum emotion strength score for speaker A",
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
        emotion_strength_threshold=args.emotion_strength_threshold,
        min_a_emotion_strength=args.min_a_emotion_strength,
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
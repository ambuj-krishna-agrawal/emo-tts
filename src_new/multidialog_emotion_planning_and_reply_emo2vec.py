#!/usr/bin/env python3
from __future__ import annotations
import json
import logging
import sys
import time
import argparse
import threading
import traceback
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from collections import Counter
import random

# ------------------- Setup ----------------------
def setup_logging(log_file="emotion_classification.log"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("emotion_classifier")

# ------------------- Constants --------------------
PAIRS_METADATA = Path("/data/group_data/starlight/gpa/tts/multidialog_sds_pairs/pairs_metadata.json")
OUTPUT_DIR = Path("data/group_data/starlight/gpa/tts/multidialog_emotion_planning_emotion2vec")
MAX_TOKENS_GEN = 40
TEMPERATURE = 0.0
MAX_WORKERS = 4
BATCH_SIZE = 8
RANDOM_SEED = 42

ALLOWED_EMOTIONS = {
    "Happy", "Sad", "Angry", "Neutral",
    "Surprise", "Disgust", "Fear", "Excited"
}

EMOTION_MAPPING = {
    # standard eight emotions
    "neutral": "Neutral",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "surprise": "Surprise",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happy",
    "excited": "Excited",
    "curious to dive deeper": "Excited"
}

# ------------------- Data Classes --------------------
@dataclass
class EntryResult:
    """Result of processing a single dialogue entry"""
    pair_id: str
    history: str
    speaker_name: str
    model_emotion: str
    emotion_steered_reply: str
    baseline_reply: str
    reference_emotion: str
    reference_response: str
    timestamp: str
    emotion2vec_scores: Optional[List[float]] = None
    emotion2vec_labels: Optional[List[str]] = None
    processing_time: float = 0.0

# ------------------- Emotion Recognition --------------------
def run_emotion2vec(audio_paths: Dict[str, Path], 
                   output_dir: Path, 
                   logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """Run emotion2vec on audio files"""
    logger.info("Using FunASR for emotion recognition")
    return run_with_funasr(audio_paths, output_dir, logger)

def run_with_funasr(audio_paths: Dict[str, Path], 
                   output_dir: Path, 
                   logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """Process audio files with FunASR emotion2vec"""
    try:
        from funasr import AutoModel
    except ImportError as exc:
        logger.error(f"FunASR missing: {exc}")
        sys.exit(1)

    # Initialize model
    try:
        logger.info("Loading emotion2vec plus_large model")
        model = AutoModel(model="iic/emotion2vec_plus_large")
    except Exception as e:
        logger.warning(f"Failed to load plus_large model: {e}. Falling back to base_finetuned")
        model = AutoModel(model="iic/emotion2vec_base_finetuned")

    # Batch process audio files
    results = process_audio_in_batches(
        audio_paths=audio_paths,
        processor=lambda wav_path: process_audio_funasr(model, wav_path),
        logger=logger,
        desc="Processing audio with FunASR"
    )
    
    return results

def process_audio_funasr(model, wav_path: Path) -> Dict[str, Any]:
    """Process a single audio file with FunASR"""
    if not wav_path.exists():
        return {"error": f"File not found: {wav_path}"}
    
    try:
        rec = model.generate(str(wav_path), granularity="utterance", extract_embedding=False)
        
        # Normalize output
        if isinstance(rec, list) and rec and isinstance(rec[0], dict):
            rec = rec[0]
        
        scores = rec.get("scores") or rec.get("score")
        labels = rec.get("labels") or rec.get("label")
        
        if scores is None or labels is None:
            return {"error": "Missing scores or labels in model output"}
        
        idx = int(np.argmax(scores))
        return {
            "scores": scores, 
            "labels": labels,
            "predicted_emotion_idx": idx,
            "predicted_emotion_score": float(scores[idx]),
            "predicted_emotion_label": labels[idx],
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

def process_audio_in_batches(audio_paths: Dict[str, Path],
                           processor: callable,
                           logger: logging.Logger,
                           desc: str = "Processing audio",
                           batch_size: int = BATCH_SIZE,
                           max_workers: int = MAX_WORKERS) -> Dict[str, Dict[str, Any]]:
    """Process audio files in batches with parallel execution"""
    results = {}
    total_files = len(audio_paths)
    
    # Create batches for processing
    items = list(audio_paths.items())
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    logger.info(f"Processing {total_files} audio files in {len(batches)} batches")
    
    with tqdm(total=total_files, desc=desc) as pbar:
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_item = {
                    executor.submit(process_audio_item, pair_id, wav_path, processor): 
                    (pair_id, wav_path) for pair_id, wav_path in batch
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_item):
                    pair_id, result = future.result()
                    results[pair_id] = result
                    
                    if "error" in result:
                        logger.warning(f"Error processing {pair_id}: {result['error']}")
                    
                    pbar.update(1)
    
    # Filter out errors for downstream processing
    success_count = sum(1 for r in results.values() if "error" not in r)
    logger.info(f"Successfully processed {success_count}/{total_files} audio files")
    
    return {k: v for k, v in results.items() if "error" not in v}

def process_audio_item(pair_id: str, wav_path: Path, processor: callable) -> Tuple[str, Dict[str, Any]]:
    """Process a single audio item and return (pair_id, result)"""
    try:
        result = processor(wav_path)
        return pair_id, result
    except Exception as e:
        return pair_id, {"error": str(e), "traceback": traceback.format_exc()}

def map_emotion2vec_to_allowed(emotion_label: str) -> str:
    """Map emotion2vec label to standardized format"""
    # Convert to lowercase for case-insensitive matching
    lower_label = emotion_label.lower()
    
    # Direct mapping for exact matches
    mapping = {
        "neutral": "Neutral",
        "happy": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "surprise": "Surprise",
        "fear": "Fear",
        "disgust": "Disgust",
        "joy": "Happy",
        "excited": "Excited",
        "happiness": "Happy"
    }
    
    if lower_label in mapping:
        return mapping[lower_label]
    
    # For more complex mappings
    if "excite" in lower_label:
        return "Excited"
    if "happ" in lower_label or "joy" in lower_label:
        return "Happy"
    if "ang" in lower_label or "mad" in lower_label:
        return "Angry"
    if "sad" in lower_label or "unhapp" in lower_label or "depress" in lower_label:
        return "Sad"
    if "surp" in lower_label or "shock" in lower_label:
        return "Surprise"
    if "disg" in lower_label or "revul" in lower_label:
        return "Disgust"
    if "fear" in lower_label or "afraid" in lower_label or "scar" in lower_label:
        return "Fear"
    
    # Default to Neutral if no match
    return "Neutral"

# ------------------- Text Generation --------------------
# ------------------- Text Generation --------------------
from typing import List, Dict

def build_generation_prompt(history: str, emotion: str, speaker_name: str) -> List[Dict[str, str]]:
    """
    Build a prompt that asks the model to write the next utterance in a
    DailyDialogue‑style two‑person conversation, conveying the target emotion.
    The reply must be ≤15 words and should NOT start with a speaker name.
    """
    shots: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are writing the next utterance in a short, casual dialogue (DailyDialogue style).\n"
                f"The reply should clearly express the emotion: {emotion}.\n"
                "Keep it under 15 words.\n"
                "Do NOT start with the speaker’s name; just write the line itself."
            )
        }
    ]

    fewshot_examples = [
        # (history, reply)
        ("I didn't expect you to come!\nEMOTION=Surprise",
         "Wow, you're here? What a surprise!"),
        ("The lights just went out...\nEMOTION=Fear",
         "It's so dark—this feels creepy."),
        ("I got the job offer today!\nEMOTION=Happy",
         "That's amazing news, congratulations!"),
        ("I failed the exam again.\nEMOTION=Sad",
         "I'm really sorry. That hurts."),
        ("You broke my phone.\nEMOTION=Angry",
         "Seriously? That was brand new!"),
        ("This smells awful.\nEMOTION=Disgust",
         "Ugh, please move it away."),
        ("We can finally travel next month!\nEMOTION=Excited",
         "Yes! I can't wait for the trip!"),
        ("I'll call you later about the plan.\nEMOTION=Neutral",
         "Sure, talk to you later.")
    ]

    for hist, reply in fewshot_examples:
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})

    # User turn with the real dialogue context
    shots.append({
        "role": "user",
        "content": f"{history}\nEMOTION={emotion}"
    })
    return shots


def build_baseline_prompt(history: str, speaker_name: str) -> List[Dict[str, str]]:
    """
    Build a neutral baseline prompt (DailyDialogue style).
    The reply must be ≤15 words and should NOT start with a speaker name.
    """
    shots: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are writing the next utterance in a short, casual dialogue (DailyDialogue style).\n"
                "The reply should express a NEUTRAL emotion.\n"
                "Keep it under 15 words.\n"
                "Do NOT start with the speaker’s name; just write the line itself."
            )
        }
    ]

    fewshot_examples = [
        ("I'll be at the library tonight.",
         "Okay, see you there."),
        ("Can you send me the file?",
         "Sure, I'll email it now."),
        ("The meeting starts at nine.",
         "Got it, I'll be on time."),
        ("It's raining again.",
         "Yeah, remember your umbrella."),
        ("I'm cooking pasta for dinner.",
         "Sounds good, enjoy.")
    ]

    for hist, reply in fewshot_examples:
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})

    # User turn with real dialogue context
    shots.append({
        "role": "user",
        "content": history
    })
    return shots


def generate_responses_batch(router, alias: str, 
                            entries: List[Dict[str, Any]], 
                            emotion2vec_results: Dict[str, Dict[str, Any]],
                            logger: logging.Logger) -> List[Dict[str, Any]]:
    """Generate responses for multiple entries in optimized batches"""
    # Prepare all prompts first
    steered_prompts = []
    baseline_prompts = []
    
    for entry in entries:
        pair_id = entry["pair_id"]
        
        # Get emotion from emotion2vec
        e2v_result = emotion2vec_results.get(pair_id, {})
        if not e2v_result:
            model_emotion = "Neutral"
        else:
            raw_emotion = e2v_result.get("predicted_emotion_label", "neutral")
            model_emotion = map_emotion2vec_to_allowed(raw_emotion)
        
        # Create prompts
        steered_prompts.append((
            f"{pair_id}_steered", 
            build_generation_prompt(entry["text_A"], model_emotion, entry["speaker_B"])
        ))
        
        baseline_prompts.append((
            f"{pair_id}_baseline", 
            build_baseline_prompt(entry["text_A"], entry["speaker_B"])
        ))
    
    # Generate all responses (steered first, then baseline)
    logger.info(f"Generating {len(steered_prompts)} emotion-steered responses")
    
    steered_results = batch_generate(
        router=router,
        alias=alias,
        prompts=steered_prompts,
        max_tokens=MAX_TOKENS_GEN,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        desc="Generating emotion-steered responses"
    )
    
    logger.info(f"Generating {len(baseline_prompts)} baseline neutral responses")
    
    baseline_results = batch_generate(
        router=router,
        alias=alias,
        prompts=baseline_prompts,
        max_tokens=MAX_TOKENS_GEN,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        desc="Generating baseline responses"
    )
    
    # Combine results into final entries
    results = []
    
    for entry in entries:
        pair_id = entry["pair_id"]
        
        # Get emotion from emotion2vec
        e2v_result = emotion2vec_results.get(pair_id, {})
        if not e2v_result:
            model_emotion = "Neutral"
            emotion2vec_scores = None
            emotion2vec_labels = None
        else:
            raw_emotion = e2v_result.get("predicted_emotion_label", "neutral")
            model_emotion = map_emotion2vec_to_allowed(raw_emotion)
            emotion2vec_scores = e2v_result.get("scores")
            emotion2vec_labels = e2v_result.get("labels")
        
        # Get generated responses
        steered_response = steered_results.get(f"{pair_id}_steered", f"{entry['speaker_B']}: (no response)")
        baseline_response = baseline_results.get(f"{pair_id}_baseline", f"{entry['speaker_B']}: (no response)")
        
        # Create result entry
        result = EntryResult(
            pair_id=pair_id,
            history=entry["text_A"],
            speaker_name=entry["speaker_B"],
            model_emotion=model_emotion,
            emotion_steered_reply=steered_response,
            baseline_reply=baseline_response,
            reference_emotion=map_emotion(entry.get("emotion_B", "")),
            reference_response=entry["text_B"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            emotion2vec_scores=emotion2vec_scores,
            emotion2vec_labels=emotion2vec_labels
        )
        
        results.append(vars(result))
    
    return results

def batch_generate(router, alias: str, prompts: List[Tuple[str, List[Dict[str, str]]]], 
                  max_tokens: int, temperature: float,
                  batch_size: int = 4, max_workers: int = 4, 
                  desc: str = "Generating responses") -> Dict[str, str]:
    """
    Generate responses for multiple prompts in optimized batches
    
    Args:
        router: LLM router/client
        alias: Model alias
        prompts: List of (prompt_id, prompt_content) tuples
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for processing
        max_workers: Max concurrent workers
        desc: Description for progress bar
        
    Returns:
        Dictionary mapping prompt_ids to generated responses
    """
    results = {}
    
    # Create batches
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    
    with tqdm(total=len(prompts), desc=desc) as pbar:
        for batch in batches:
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                
                for prompt_id, prompt_content in batch:
                    future = executor.submit(
                        generate_single_response,
                        router=router,
                        alias=alias,
                        prompt=prompt_content,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    futures[future] = prompt_id
                
                # Collect results as they complete
                for future in as_completed(futures):
                    prompt_id = futures[future]
                    try:
                        response = future.result()
                        results[prompt_id] = response
                    except Exception as e:
                        results[prompt_id] = f"ERROR: {str(e)}"
                    
                    pbar.update(1)
    
    return results

def generate_single_response(router, alias: str, prompt: List[Dict[str, str]], 
                            max_tokens: int, temperature: float) -> str:
    """Generate a single response with error handling"""
    try:
        response = router.inference_call(
            alias, prompt, max_tokens=max_tokens, temperature=temperature
        ).strip()
        
        # Take only the first line
        response = response.splitlines()[0] if response.splitlines() else response
        return response
    except Exception as e:
        raise RuntimeError(f"Generation error: {str(e)}")
    
def remap_and_downsample(pairs: list) -> list:
    """
    1) Map each pair's B-side emotion_B -> standard label,
    2) Count frequencies,
    3) If "Excited" is larger than the next-most-common class, downsample "Excited" to that size.
    """
    # 1) Remap all reference emotions
    for p in pairs:
        raw = p.get("emotion_B", "")
        p["reference_emotion"] = map_emotion(raw)

    # 2) Count frequencies of each reference_emotion
    freqs = Counter(p["reference_emotion"] for p in pairs)
    excited_count = freqs.get("Excited", 0)
    # Find the largest count among all other classes
    other_counts = [cnt for emo, cnt in freqs.items() if emo != "Excited"]
    if not other_counts:
        # Nothing to compare against
        return pairs
    max_other = max(other_counts)

    # 3) Downsample if needed
    if excited_count > max_other:
        random.seed(RANDOM_SEED)
        # Separate excited vs others
        excited = [p for p in pairs if p["reference_emotion"] == "Excited"]
        others = [p for p in pairs if p["reference_emotion"] != "Excited"]
        # Sample exactly max_other from excited
        sampled_excited = random.sample(excited, max_other)
        # Combine and shuffle
        result = others + sampled_excited
        random.shuffle(result)
        return result

    # If no downsampling needed, just return mapped pairs
    return pairs


# ------------------- Utilities --------------------
def map_emotion(raw: str) -> str:
    """Map a raw emotion string to one of the eight standardized labels."""
    key = raw.strip().lower()
    if key in EMOTION_MAPPING:
        return EMOTION_MAPPING[key]
    # Fallback: title-case whatever remains
    return raw.strip().title()

def load_pairs(file_path: Path) -> List[Dict[str, Any]]:
    """Load dialogue pairs from JSON file"""
    try:
        data = json.loads(file_path.read_text())
        return data
    except Exception as e:
        print(f"Error loading pairs from {file_path}: {e}")
        raise

def prepare_audio_paths(pairs: List[Dict[str, Any]]) -> Dict[str, Path]:
    """Extract audio paths from pairs data"""
    audio_paths = {}
    for pair in pairs:
        pair_id = pair["pair_id"]
        audio_path = pair.get("audio_path_A")
        if audio_path:
            audio_paths[pair_id] = Path(audio_path)
    return audio_paths

def write_results(results: List[Dict[str, Any]], output_file: Path):
    """Write results to JSONL file"""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def get_model_configs():
    """Get model configurations"""
    return {
        "llama_3_70b_q4": {
            "model_id": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
            "base_url": "http://babel-7-25:8081/v1",
            "is_chat": False,
            "template_builder": None
        },
        "llama_3_3b_q4": {
            "model_id": "AMead10/Llama-3.2-3B-Instruct-AWQ",
            "base_url": "http://babel-0-23:8083/v1",
            "is_chat": False,
            "template_builder": None
        },
        "mistral_7b_q4": {
            "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            "base_url": "http://babel-0-23:8082/v1",
            "is_chat": False,
            "template_builder": None
        }
    }

def create_inference_client():
    """Create and configure the inference client"""
    from src_new.vllm_router import RawTemplateBuilder, VLLMChatCompletion
    cfgs = get_model_configs()
    for cfg in cfgs.values():
        cfg["template_builder"] = RawTemplateBuilder()
    return VLLMChatCompletion(cfgs)

# ------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Run emotion classification with batched processing")
    parser.add_argument("--input", "-i", help="Path to pairs metadata JSON file", default=str(PAIRS_METADATA))
    parser.add_argument("--output-dir", "-o", help="Directory to store results", default=str(OUTPUT_DIR))
    parser.add_argument("--model", "-m", default="llama_3_70b_q4", help="Model alias to use")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help="Maximum concurrent workers")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--cached-emotions", help="Path to cached emotion2vec results")
    args = parser.parse_args()

    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logging(output_dir / "emotion_classification.log")
    
    # Log configuration
    logger.info(f"Starting with configuration:")
    logger.info(f"  - Input file: {args.input}")
    logger.info(f"  - Output directory: {output_dir}")
    logger.info(f"  - Model: {args.model}")
    logger.info(f"  - Workers: {args.workers}")
    logger.info(f"  - Batch size: {args.batch_size}")
    
    # Load pairs data
    logger.info(f"Loading pairs from {args.input}")
    pairs = load_pairs(Path(args.input))
    logger.info(f"Loaded {len(pairs)} pairs")
    
    # Remap and downsample to balance emotion distribution
    original_count = len(pairs)
    pairs = remap_and_downsample(pairs)
    logger.info(f"After remapping and downsampling: {len(pairs)} pairs (originally {original_count})")
    
    # Load or generate emotion2vec results
    emotion2vec_results = {}
    if args.cached_emotions and Path(args.cached_emotions).exists():
        logger.info(f"Loading cached emotion2vec results from {args.cached_emotions}")
        with open(args.cached_emotions, 'r') as f:
            emotion2vec_results = json.load(f)
    else:
        # Extract audio paths and run emotion2vec
        audio_paths = prepare_audio_paths(pairs)
        logger.info(f"Running emotion2vec on {len(audio_paths)} audio files")
        
        emotion2vec_results = run_emotion2vec(
            audio_paths=audio_paths,
            output_dir=output_dir,
            logger=logger
        )
        
        # Cache results
        cache_path = output_dir / "emotion2vec_results.json"
        with open(cache_path, 'w') as f:
            json.dump(emotion2vec_results, f, indent=2)
        logger.info(f"Cached emotion2vec results to {cache_path}")
    
    # Initialize the LLM client
    logger.info(f"Initializing LLM client for {args.model}")
    client = create_inference_client()
    
    # Generate responses in batches
    logger.info(f"Generating responses with {args.model}")
    results = generate_responses_batch(
        router=client,
        alias=args.model,
        entries=pairs,
        emotion2vec_results=emotion2vec_results,
        logger=logger
    )
    
    # Write results to file
    output_file = output_dir / f"{args.model}_results.jsonl"
    write_results(results, output_file)
    logger.info(f"Wrote {len(results)} results to {output_file}")
    
    # Calculate and write statistics
    emotion_counts = {}
    agreement_count = 0
    
    for result in results:
        # Track predicted emotions
        emotion = result.get("model_emotion", "Unknown")
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 0
        emotion_counts[emotion] += 1
        
        # Check agreement with reference emotion
        if result.get("model_emotion") == result.get("reference_emotion"):
            agreement_count += 1
    
    stats = {
        "total_processed": len(results),
        "emotion_distribution": emotion_counts,
        "agreement_with_reference": agreement_count,
        "agreement_percentage": (agreement_count / len(results) * 100) if results else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    stats_path = output_dir / f"{args.model}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Wrote statistics to {stats_path}")
    logger.info(f"Processing complete. Results: {len(results)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
#!/usr/bin/env python3
"""
End‑to‑end pipeline for planning the emotion of a reply and then generating the
reply itself.  Compared with the previous version, **all audio‑based emotion2vec
logic has been removed**.  Instead, we directly predict the target emotion for
Speaker B from Speaker A’s last utterance **and** the emotion that Speaker A is
feeling (``emotion_A`` in the pairs metadata).  The prediction is done with an
LLM classification prompt that contains multi‑shot examples.  Once the target
emotion is decided, we construct a generation prompt to produce a ≤15‑word
reply that conveys that emotion, exactly as before.  A neutral baseline reply
is also produced for comparison.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

# ------------------- Logging --------------------

def setup_logging(log_file: str | Path = "emotion_classification.log") -> logging.Logger:
    """Configure python‐logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("emotion_reply")


# ------------------- Constants --------------------

PAIRS_METADATA = Path(
    "/data/group_data/starlight/gpa/tts/multidialog_sds_pairs/pairs_metadata.json"
)
OUTPUT_DIR = Path("data/group_data/starlight/gpa/tts/multidialog_emotion_planning_2")
MAX_TOKENS_GEN = 40  # for the *generation* step
MAX_TOKENS_CLASS = 4  # we only need a single word for the classification step
TEMPERATURE = 0.0
MAX_WORKERS = 4
BATCH_SIZE = 8
RANDOM_SEED = 42

ALLOWED_EMOTIONS: set[str] = {
    "Happy",
    "Sad",
    "Angry",
    "Neutral",
    "Surprise",
    "Disgust",
    "Fear",
    "Excited",
}

EMOTION_MAPPING: dict[str, str] = {
    # explicit standardisation map (keys are lowercase)
    "neutral": "Neutral",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "surprise": "Surprise",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happy",
    "excited": "Excited",
}

# ------------------- Data Classes --------------------


@dataclass
class EntryResult:
    """JSON‑serialisable container for one processed dialogue pair."""

    pair_id: str
    history: str
    speaker_name: str
    target_emotion: str  # emotion the model *decided* to convey
    emotion_steered_reply: str
    baseline_reply: str
    reference_emotion: str  # the gold label for Speaker B (if present)
    reference_response: str
    timestamp: str


# ------------------- Utility functions --------------------


def map_emotion(raw: str) -> str:
    """Map a raw emotion string to one of the eight standardised labels."""
    key = raw.strip().lower()
    if key in EMOTION_MAPPING:
        return EMOTION_MAPPING[key]
    # Fallback: title‑case whatever remains and hope it is valid
    return raw.strip().title()


# ------------------- Prompt builders --------------------


def build_classification_prompt(history: str, emotion_a: str) -> List[Dict[str, str]]:
    """Construct a *classification* prompt that asks for the best reply emotion.

    The LLM must answer with **exactly one** word chosen from ``ALLOWED_EMOTIONS``.
    We supply several DailyDialogue‑style few‑shot examples to guide the choice.
    """

    examples: list[tuple[str, str, str]] = [
        (
            "I failed the exam again.",
            "Sad",
            "Sad",  # empathetic
        ),
        (
            "I got the job offer today!",
            "Happy",
            "Excited",  # share their joy enthusiastically
        ),
        (
            "You broke my phone.",
            "Angry",
            "Neutral",  # stay calm to de‑escalate
        ),
        (
            "There's a strange noise downstairs.",
            "Fear",
            "Neutral",  # reassure
        ),
        (
            "We can finally travel next month!",
            "Excited",
            "Excited",  # mirror excitement
        ),
    ]

    shots: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an assistant deciding what emotion Speaker B should express in their"
                " next response.  Choose **one** word from this list exactly as written: "
                f"{', '.join(sorted(ALLOWED_EMOTIONS))}. Return only that word."
            ),
        }
    ]

    for utt, emo_a, emo_b in examples:
        user_msg = f"UTTERANCE: {utt}\nEMOTION_A={emo_a}"
        shots.append({"role": "user", "content": user_msg})
        shots.append({"role": "assistant", "content": emo_b})

    # real example
    shots.append(
        {
            "role": "user",
            "content": f"UTTERANCE: {history}\nEMOTION_A={emotion_a}",
        }
    )
    return shots



def build_generation_prompt(history: str, emotion: str) -> List[Dict[str, str]]:
    """Prompt to *write* Speaker B's reply conveying the chosen ``emotion``."""

    fewshot_examples: list[tuple[str, str]] = [
        ("I didn't expect you to come!\nEMOTION=Surprise", "Wow, you're here? What a surprise!"),
        (
            "The lights just went out...\nEMOTION=Fear",
            "It's so dark—this feels creepy.",
        ),
        (
            "I got the job offer today!\nEMOTION=Happy",
            "That's amazing news, congratulations!",
        ),
        (
            "I failed the exam again.\nEMOTION=Sad",
            "I'm really sorry. That hurts.",
        ),
        (
            "You broke my phone.\nEMOTION=Angry",
            "Seriously? That was brand new!",
        ),
        (
            "This smells awful.\nEMOTION=Disgust",
            "Ugh, please move it away.",
        ),
        (
            "We can finally travel next month!\nEMOTION=Excited",
            "Yes! I can't wait for the trip!",
        ),
        ("I'll call you later about the plan.\nEMOTION=Neutral", "Sure, talk to you later."),
    ]

    shots: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are writing Speaker B's next utterance in a short DailyDialogue‑style"
                " conversation.  The reply must *clearly* convey the target emotion: "
                f"{emotion}.  Keep it under 15-20 words and **do NOT** start with a speaker"
                " name—just write the line itself."
            ),
        }
    ]

    for hist, reply in fewshot_examples:
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})

    shots.append({"role": "user", "content": f"{history}\nEMOTION={emotion}"})
    return shots



def build_baseline_prompt(history: str) -> List[Dict[str, str]]:
    """Prompt for a *neutral* baseline reply."""

    fewshot_examples: list[tuple[str, str]] = [
        ("I'll be at the library tonight.", "Okay, see you there."),
        ("Can you send me the file?", "Sure, I'll email it now."),
        ("The meeting starts at nine.", "Got it, I'll be on time."),
        ("It's raining again.", "Yeah, remember your umbrella."),
        ("I'm cooking pasta for dinner.", "Sounds good, enjoy."),
    ]

    shots: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are writing Speaker B's next utterance in a short DailyDialogue‑style"
                " conversation.  The reply should be **NEUTRAL**. Keep it under 15 words"
                " and do NOT start with a speaker name."
            ),
        }
    ]

    for hist, reply in fewshot_examples:
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})

    shots.append({"role": "user", "content": history})
    return shots


# ------------------- LLM helpers --------------------


def batch_generate(
    router,
    alias: str,
    prompts: List[Tuple[str, List[Dict[str, str]]]],
    max_tokens: int,
    temperature: float,
    batch_size: int = 4,
    max_workers: int = 4,
    desc: str = "Generating",
) -> Dict[str, str]:
    """Generate responses for many prompts in (concurrent) batches."""

    results: dict[str, str] = {}
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

    with tqdm(total=len(prompts), desc=desc) as pbar:
        for batch in batches:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        generate_single_response,
                        router,
                        alias,
                        prompt,
                        max_tokens,
                        temperature,
                    ): prompt_id
                    for prompt_id, prompt in batch
                }

                for future in as_completed(futures):
                    prompt_id = futures[future]
                    try:
                        resp = future.result()
                        results[prompt_id] = resp
                    except Exception as err:
                        results[prompt_id] = f"ERROR: {err}"
                    pbar.update(1)
    return results



def generate_single_response(router, alias: str, prompt: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
    """Wrapper around ``router.inference_call`` with basic trimming."""

    response: str = (
        router.inference_call(alias, prompt, max_tokens=max_tokens, temperature=temperature).strip()
    )
    # Use only the first line to avoid trailing explanations
    return response.splitlines()[0] if response.splitlines() else response


# ------------------- Core pipeline --------------------


def generate_responses_batch(
    router,
    alias: str,
    entries: List[Dict[str, Any]],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Main batched pipeline: classify target emotion ➜ generate replies."""

    # ---------- 1) Predict target emotion ----------
    classification_prompts: list[tuple[str, List[dict[str, str]]]] = []
    for entry in entries:
        pair_id = entry["pair_id"]
        emotion_a_raw = entry.get("emotion_A", "")
        emotion_a = map_emotion(emotion_a_raw) if emotion_a_raw else "Neutral"
        classification_prompts.append(
            (f"{pair_id}_class", build_classification_prompt(entry["text_A"], emotion_a))
        )

    logger.info(f"Predicting target emotions for {len(classification_prompts)} pairs")
    classification_results = batch_generate(
        router=router,
        alias=alias,
        prompts=classification_prompts,
        max_tokens=MAX_TOKENS_CLASS,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        desc="Classifying emotions",
    )

    # ---------- 2) Build generation tasks ----------
    steered_prompts: list[tuple[str, List[dict[str, str]]]] = []
    baseline_prompts: list[tuple[str, List[dict[str, str]]]] = []

    for entry in entries:
        pair_id = entry["pair_id"]
        target_emotion_raw = classification_results.get(f"{pair_id}_class", "Neutral")
        target_emotion = map_emotion(target_emotion_raw)
        if target_emotion not in ALLOWED_EMOTIONS:
            target_emotion = "Neutral"

        steered_prompts.append(
            (f"{pair_id}_steered", build_generation_prompt(entry["text_A"], target_emotion))
        )
        baseline_prompts.append((f"{pair_id}_baseline", build_baseline_prompt(entry["text_A"])) )

    # ---------- 3) Generate replies ----------
    logger.info(f"Generating {len(steered_prompts)} emotion‑steered replies")
    steered_results = batch_generate(
        router=router,
        alias=alias,
        prompts=steered_prompts,
        max_tokens=MAX_TOKENS_GEN,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        desc="Generating steered replies",
    )

    logger.info(f"Generating {len(baseline_prompts)} baseline replies")
    baseline_results = batch_generate(
        router=router,
        alias=alias,
        prompts=baseline_prompts,
        max_tokens=MAX_TOKENS_GEN,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        desc="Generating baseline replies",
    )

    # ---------- 4) Consolidate outputs ----------
    results: list[dict[str, Any]] = []
    for entry in entries:
        pair_id = entry["pair_id"]
        target_emotion_raw = classification_results.get(f"{pair_id}_class", "Neutral")
        target_emotion = map_emotion(target_emotion_raw)
        if target_emotion not in ALLOWED_EMOTIONS:
            target_emotion = "Neutral"

        result = EntryResult(
            pair_id=pair_id,
            history=entry["text_A"],
            speaker_name=entry["speaker_B"],
            target_emotion=target_emotion,
            emotion_steered_reply=steered_results.get(
                f"{pair_id}_steered", f"{entry['speaker_B']}: (no response)"
            ),
            baseline_reply=baseline_results.get(
                f"{pair_id}_baseline", f"{entry['speaker_B']}: (no response)"
            ),
            reference_emotion=map_emotion(entry.get("emotion_B", "")),
            reference_response=entry["text_B"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        results.append(vars(result))

    return results


# ------------------- Dataset helpers --------------------


def load_pairs(file_path: Path) -> List[Dict[str, Any]]:
    """Load the pairs metadata JSON file."""

    try:
        return json.loads(file_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Error loading pairs from {file_path}: {exc}") from exc



def remap_and_downsample(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map ``emotion_B`` to standard labels and optionally downsample *Excited*.

    1. ``emotion_B`` ➜ ``reference_emotion`` (standard label)
    2. Compute class frequencies.
    3. If *Excited* has more instances than the next‑largest class, randomly
       downsample it to that size (to balance the dataset for evaluation).
    """

    # 1) remap reference emotions
    for p in pairs:
        p["reference_emotion"] = map_emotion(p.get("emotion_B", ""))

    # 2) frequency counts
    freqs = Counter(p["reference_emotion"] for p in pairs)
    excited_count = freqs.get("Excited", 0)
    other_counts = [cnt for emo, cnt in freqs.items() if emo != "Excited"]

    if not other_counts:
        return pairs  # nothing to downsample

    max_other = max(other_counts)

    # 3) downsample if needed
    if excited_count > max_other:
        random.seed(RANDOM_SEED)
        excited = [p for p in pairs if p["reference_emotion"] == "Excited"]
        others = [p for p in pairs if p["reference_emotion"] != "Excited"]
        sampled_excited = random.sample(excited, max_other)
        combined = others + sampled_excited
        random.shuffle(combined)
        return combined

    return pairs


# ------------------- Model/router setup --------------------


def get_model_configs() -> dict[str, dict[str, Any]]:
    """Return model config dictionary for the VLLM router."""

    return {
        "llama_3_70b_q4": {
            "model_id": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
            "base_url": "http://babel-7-25:8081/v1",
            "is_chat": False,
            "template_builder": None,
        },
        "llama_3_3b_q4": {
            "model_id": "AMead10/Llama-3.2-3B-Instruct-AWQ",
            "base_url": "http://babel-0-23:8083/v1",
            "is_chat": False,
            "template_builder": None,
        },
        "mistral_7b_q4": {
            "model_id": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            "base_url": "http://babel-0-23:8082/v1",
            "is_chat": False,
            "template_builder": None,
        },
    }



def create_inference_client():
    """Instantiate the VLLM router client with raw prompting."""

    from src_new.vllm_router import RawTemplateBuilder, VLLMChatCompletion

    cfgs = get_model_configs()
    for cfg in cfgs.values():
        cfg["template_builder"] = RawTemplateBuilder()
    return VLLMChatCompletion(cfgs)

def filter_top_n_by_length(pairs: list[dict[str, Any]], n: int = 2000) -> list[dict[str, Any]]:
    """Filter to keep only the top N pairs based on text length (longer texts first).
    
    Args:
        pairs: List of dialogue pair dictionaries
        n: Number of pairs to keep (default: 2000)
        
    Returns:
        Filtered list with the top N longest pairs
    """
    # Sort pairs by combined length of text_A (descending order)
    sorted_pairs = sorted(
        pairs, 
        key=lambda p: len(p.get("text_A", "")), 
        reverse=True
    )
    
    # Take the top N pairs
    return sorted_pairs[:n]

# ------------------- Main entry point --------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan reply emotion (classification) and generate reply (llm)"
    )
    parser.add_argument("--input", "-i", default=str(PAIRS_METADATA), help="Pairs metadata JSON")
    parser.add_argument("--output-dir", "-o", default=str(OUTPUT_DIR), help="Directory for outputs")
    parser.add_argument("--model", "-m", default="llama_3_70b_q4", help="Model alias to use")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help="Concurrent workers")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE, help="Batch size")
    # Add the new filter argument
    parser.add_argument("--filter-top", "-f", type=int, default=2000, 
                        help="Filter to process only top N longest pairs (0 = process all)")
    args = parser.parse_args()

    # ---- setup ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logging(output_dir / "emotion_reply.log")

    logger.info("Configuration: %s", vars(args))

    # ---- data ----
    logger.info("Loading pairs from %s", args.input)
    pairs = load_pairs(Path(args.input))
    logger.info("Loaded %d pairs", len(pairs))

    # Apply filtering if requested
    if args.filter_top > 0:
        original_count = len(pairs)
        pairs = filter_top_n_by_length(pairs, args.filter_top)
        logger.info(f"Filtered to top {args.filter_top} pairs by text length (from {original_count})")

    pairs = remap_and_downsample(pairs)
    logger.info("After downsampling: %d pairs", len(pairs))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        sys.exit(1)

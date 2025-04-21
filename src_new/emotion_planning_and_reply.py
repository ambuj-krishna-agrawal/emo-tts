#!/usr/bin/env python3
"""
End‑to‑end pipeline for *planning* the emotion of Speaker B’s reply and then
*generating* that reply on the MELD‑SDS Friends dataset.

**What changed in this revision (2025‑04‑21):**

*   🔄  *Classification prompt*: we no longer ask the LLM to recognise the
    emotion of Speaker A’s line (``emotion_A`` is already in the dataset).
    Instead we ask it to *choose the most appropriate target emotion for
    Speaker B’s response*, given **(a)** Speaker A’s utterance *and* **(b)**
    Speaker A’s emotion label.
*   🆕  ``build_target_emotion_prompt`` now takes both the utterance and
    ``emotion_A`` and returns a one‑word label from ``ALLOWED_EMOTIONS``.
    Few‑shot examples were rewritten with MELD/Friends‑style lines to show the
    mapping ➜ target emotion.
*   🛠️  ``process_example`` obtains ``emotion_A`` from the metadata and feeds
    it into the new prompt builder.
*   All other logic (batched parallelism, thread‑safe writers, statistics, etc.)
    is unchanged.
"""
from __future__ import annotations

# ------------------- stdlib imports --------------------
import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

# ------------------- logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("emotion_classification.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ------------------- constants --------------------
PAIRS_METADATA = Path("meld_audio_sds_pairs_custom_balanced/pairs_metadata.json")
OUTPUT_DIR = Path("emotion_planning_results_custom_balanced")
MAX_TOKENS_CLASS = 4   # only need a single word
MAX_TOKENS_GEN = 40
TEMPERATURE = 0.0
MAX_WORKERS = 4

ALLOWED_EMOTIONS: Set[str] = {
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
    "neutral": "Neutral",
    "surprise": "Surprise",
    "fear": "Fear",
    "sad": "Sad",
    "angry": "Angry",
    "disgust": "Disgust",
    "happy": "Happy",
    "joy": "Happy",
    "excited": "Excited",
}

# ------------------- dataclasses --------------------


@dataclass
class EntryResult:
    pair_id: str
    history: str
    speaker_name: str
    model_emotion: str  # the chosen target emotion for the reply
    emotion_steered_reply: str
    baseline_reply: str
    reference_emotion: str  # gold emotion_B (for evaluation)
    reference_response: str
    timestamp: str


# ==================================================
#                 PROMPT BUILDERS
# ==================================================

def build_target_emotion_prompt(history: str, emotion_a: str) -> List[Dict[str, str]]:
    """Prompt the LLM to *choose* Speaker B’s reply emotion.

    It must reply with **exactly one** token that is a member of
    ``ALLOWED_EMOTIONS``.
    """

    allowed = ", ".join(sorted(ALLOWED_EMOTIONS))
    shots: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You decide what emotion Speaker B should express in their reply. "
                "Use the dialogue line and the emotion Speaker A is feeling. "
                "Return **one** word only, chosen from: "
                f"{allowed}. No extra text."
            ),
        }
    ]

    fewshot_examples: list[tuple[str, str, str]] = [
        (
            "I failed the exam again.",
            "Sad",
            "Sad",  # empathetic reply
        ),
        (
            "I got the job offer today!",
            "Happy",
            "Excited",  # share the joy
        ),
        (
            "You broke my phone.",
            "Angry",
            "Neutral",  # stay calm / de‑escalate
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
        (
            "This smells awful.",
            "Disgust",
            "Disgust",  # share disgust
        ),
        (
            "I didn't expect you to come!",
            "Surprise",
            "Surprise",  # match surprise
        ),
        (
            "It's raining again.",
            "Neutral",
            "Neutral",  # stay neutral
        ),
    ]

    for utt, emo_a, emo_b in fewshot_examples:
        user_msg = f"UTTERANCE: {utt}\nEMOTION_A={emo_a}"
        shots.append({"role": "user", "content": user_msg})
        shots.append({"role": "assistant", "content": emo_b})

    # Real example
    shots.append({"role": "user", "content": f"UTTERANCE: {history}\nEMOTION_A={emotion_a}"})
    return shots



def build_generation_prompt(history: str, emotion: str, speaker_name: str) -> List[Dict[str, str]]:
    """Prompt to actually *write* Speaker B’s next line conveying ``emotion``."""

    shots: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Write the next line in a Friends‑style scene. It must convey "
                f"the emotion **{emotion}**. Start with the speaker name and a colon. "
                "Keep it under 15 words."
            ),
        }
    ]

    fewshot_examples: list[tuple[str, str]] = [
        (
            "So let's talk a little bit about your duties.\nEMOTION=Surprise\nSPEAKER=Chandler",
            "Chandler: My duties? All right.",
        ),
        (
            "We can go into detail\nEMOTION=Fear\nSPEAKER=Chandler",
            "Chandler: No, don't— I beg of you!",
        ),
        (
            "I just got offered a new job.\nEMOTION=Happy\nSPEAKER=Joey",
            "Joey: Yes! Finally my acting career is taking off!",
        ),
        (
            "You ate my sandwich?!\nEMOTION=Angry\nSPEAKER=Ross",
            "Ross: That was *my* sandwich! MY SANDWICH!",
        ),
        (
            "I got the promotion!\nEMOTION=Excited\nSPEAKER=Monica",
            "Monica: Oh my god— I'm head chef!",
        ),
        (
            "What's that smell?\nEMOTION=Disgust\nSPEAKER=Rachel",
            "Rachel: Ugh, that's revolting! Get it away!",
        ),
        (
            "Let me explain the project requirements.\nEMOTION=Neutral\nSPEAKER=Monica",
            "Monica: Sure, what do we need?",
        ),
    ]

    for hist, reply in fewshot_examples:
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})

    shots.append({"role": "user", "content": f"{history}\nEMOTION={emotion}\nSPEAKER={speaker_name}"})
    return shots



def build_baseline_prompt(history: str, speaker_name: str) -> List[Dict[str, str]]:
    """Neutral baseline (under 15 words, speaker name prefix)."""

    shots: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Write the next line in a Friends‑style scene. It should be **NEUTRAL** "
                "and under 15 words. Start with the speaker name and a colon."
            ),
        }
    ]

    fewshot_examples: list[tuple[str, str]] = [
        (
            "Do you want to go to the museum?\nSPEAKER=Ross",
            "Ross: I've been there before— it was interesting.",
        ),
        (
            "I'm thinking about getting a new haircut.\nSPEAKER=Rachel",
            "Rachel: Okay, when's your appointment?",
        ),
        (
            "We need to talk about the rent.\nSPEAKER=Joey",
            "Joey: I can pay my share next week.",
        ),
    ]

    for hist, reply in fewshot_examples:
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})

    shots.append({"role": "user", "content": f"{history}\nSPEAKER={speaker_name}"})
    return shots


# ==================================================
#                UTILITY FUNCTIONS
# ==================================================

def map_emotion(raw: str) -> str:
    return EMOTION_MAPPING.get(raw.lower(), "Neutral")


def load_pairs(file_path: Path = PAIRS_METADATA) -> List[Dict[str, Any]]:
    data = json.loads(file_path.read_text())
    logger.info("Loaded %d pairs", len(data))
    return data


# ==================================================
#                SINGLE EXAMPLE PIPE
# ==================================================

def process_example(router, alias: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    pair_id = entry["pair_id"]
    history = entry["text_A"]
    speaker_name = entry["speaker_B"]

    # gold labels for evaluation
    reference_response = entry["text_B"]
    reference_emotion = map_emotion(entry.get("emotion_B", ""))

    # Speaker A's emotion from the dataset
    emotion_a = map_emotion(entry.get("emotion_A", ""))

    # 1) decide target emotion for reply
    target_prompt = build_target_emotion_prompt(history, emotion_a)
    try:
        raw_target = (
            router.inference_call(alias, target_prompt, max_tokens=MAX_TOKENS_CLASS, temperature=TEMPERATURE)
            .strip()
        )
    except Exception as exc:
        logger.error("Error classifying %s: %s", pair_id, exc)
        raw_target = "Neutral"

    model_emotion = next((e for e in ALLOWED_EMOTIONS if e.lower() in raw_target.lower()), "Neutral")

    # 2) emotion‑steered reply
    gen_prompt = build_generation_prompt(history, model_emotion, speaker_name)
    try:
        steered = (
            router.inference_call(alias, gen_prompt, max_tokens=MAX_TOKENS_GEN, temperature=TEMPERATURE)
            .strip()
        )
        steered = steered.splitlines()[0] if steered.splitlines() else steered
    except Exception as exc:
        logger.error("Error generating steered reply %s: %s", pair_id, exc)
        steered = f"{speaker_name}: (no response)"

    # 3) baseline reply
    base_prompt = build_baseline_prompt(history, speaker_name)
    try:
        baseline = (
            router.inference_call(alias, base_prompt, max_tokens=MAX_TOKENS_GEN, temperature=TEMPERATURE)
            .strip()
        )
        baseline = baseline.splitlines()[0] if baseline.splitlines() else baseline
    except Exception as exc:
        logger.error("Error generating baseline %s: %s", pair_id, exc)
        baseline = f"{speaker_name}: (no response)"

    return EntryResult(
        pair_id=pair_id,
        history=history,
        speaker_name=speaker_name,
        model_emotion=model_emotion,
        emotion_steered_reply=steered,
        baseline_reply=baseline,
        reference_emotion=reference_emotion,
        reference_response=reference_response,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    ).__dict__


# Thread‑safe writer helpers ---------------------------------------------------


class ResultWriter:
    def __init__(self, out_path: Path, progress_path: Path):
        self.out_path = out_path
        self.progress_path = progress_path
        self._out_lock = threading.Lock()
        self._prog_lock = threading.Lock()

    def write_result(self, result: Dict[str, Any]):
        with self._out_lock:
            with self.out_path.open("a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()

    def write_progress(self, pair_id: str):
        with self._prog_lock:
            with self.progress_path.open("a") as f:
                f.write(pair_id + "\n")
                f.flush()

    def write_error(self, alias: str, pair_id: str, error: Exception):
        err_path = self.out_path.parent / f"{alias}_errors.log"
        with self._out_lock:
            with err_path.open("a") as f:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} – Error with {pair_id}: {error}\n"
                )
                f.flush()


# Main per‑model worker ---------------------------------------------------------


def process_alias(alias: str, pairs: List[Dict[str, Any]], router) -> Path:
    logger.info("Starting processing for model %s", alias)

    out_path = OUTPUT_DIR / f"{alias}_results.jsonl"
    progress_path = OUTPUT_DIR / f"{alias}_progress.txt"

    # set of already processed IDs
    processed_ids: set[str] = set()
    if progress_path.exists():
        processed_ids = set(progress_path.read_text().splitlines())
        logger.info("Loaded %d previously processed IDs for %s", len(processed_ids), alias)

    writer = ResultWriter(out_path, progress_path)

    # filter pairs
    jobs = [p for p in pairs if p["pair_id"] not in processed_ids]
    logger.info("Processing %d new examples for %s", len(jobs), alias)

    success, errors = 0, 0
    pbar = tqdm(total=len(jobs), desc=alias)

    def _worker(entry):
        nonlocal success, errors
        pid = entry["pair_id"]
        try:
            result = process_example(router, alias, entry)
            writer.write_result(result)
            writer.write_progress(pid)
            with threading.Lock():
                success += 1
            return result
        except Exception as exc:
            writer.write_error(alias, pid, exc)
            with threading.Lock():
                errors += 1
        finally:
            pbar.update(1)
        return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(as_completed(ex.submit(_worker, e) for e in jobs))

    pbar.close()

    # Re‑compute global stats
    total_processed = 0
    emotion_counts: dict[str, int] = {}
    agreement_count = 0
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            d = json.loads(line)
            total_processed += 1
            emo = d.get("model_emotion", "Unknown")
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            if d.get("model_emotion") == d.get("reference_emotion"):
                agreement_count += 1

    stats = {
        "total_processed": total_processed,
        "emotion_distribution": emotion_counts,
        "agreement_with_reference": agreement_count,
        "agreement_percentage": (agreement_count / total_processed * 100) if total_processed else 0,
        "successes_this_run": success,
        "errors_this_run": errors,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    stats_path = OUTPUT_DIR / f"{alias}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Finished %s – successes: %d, errors: %d", alias, success, errors)
    return out_path


# Model/router setup -----------------------------------------------------------


def get_model_configs():
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
    from src_new.vllm_router import RawTemplateBuilder, VLLMChatCompletion

    cfgs = get_model_configs()
    for cfg in cfgs.values():
        cfg["template_builder"] = RawTemplateBuilder()
    return VLLMChatCompletion(cfgs)


# ==================================================
#                       MAIN
# ==================================================

def main():
    global OUTPUT_DIR, MAX_WORKERS

    parser = argparse.ArgumentParser(
        description="Plan reply emotion (classification) and generate reply (MELD‑SDS)"
    )
    parser.add_argument("--input", "-i", default=str(PAIRS_METADATA), help="Path to pairs metadata JSON")
    parser.add_argument("--output-dir", "-o", default=str(OUTPUT_DIR), help="Directory for outputs")
    parser.add_argument("--model", "-m", default="llama_3_70b_q4", help="Model alias to use (or 'all')")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help="Number of concurrent workers")
    parser.add_argument("--filter-top", "-f", type=int, default=0, 
                       help="Filter to process only top N longest pairs (0 = process all)")
    args = parser.parse_args()

    # Update globals from args
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    MAX_WORKERS = args.workers

    logger.info("Configuration: %s", vars(args))

    # Load pairs data
    logger.info("Loading pairs from %s", args.input)
    pairs = load_pairs(Path(args.input))
    
    # Apply filtering if requested
    if args.filter_top > 0:
        original_count = len(pairs)
        # Sort pairs by text length and take top N
        pairs = sorted(pairs, key=lambda p: len(p.get("text_A", "")), reverse=True)[:args.filter_top]
        logger.info(f"Filtered to top {args.filter_top} pairs by text length (from {original_count})")

    # Initialize LLM client
    logger.info("Initializing LLM client")
    client = create_inference_client()
    
    # Process based on model selection
    if args.model.lower() == "all":
        # Process all models
        model_aliases = list(get_model_configs().keys())
        logger.info("Processing all models: %s", ", ".join(model_aliases))
        
        for alias in model_aliases:
            process_alias(alias, pairs, client)
    else:
        # Process single model
        if args.model not in get_model_configs():
            logger.error("Unknown model alias: %s. Available models: %s", 
                         args.model, ", ".join(get_model_configs().keys()))
            sys.exit(1)
        
        process_alias(args.model, pairs, client)
    
    logger.info("Processing complete 🎉")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
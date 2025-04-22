#!/usr/bin/env python3
"""
End‑to‑end pipeline (v2.1 – 2025‑04‑21)
======================================
This revision adapts reply length **dynamically** to roughly `0.9 × len(text_A)`
(words), upgrades the emotion mapping for the dominant label *Curious to dive
deeper* → **Excited**, and keeps the neutral baseline unchanged for fair
comparison.

Key changes vs v2.0
-------------------
1. **Proportional length** – target words computed per example; the system
   prompt now says "Aim for about *N* words (±20)".  Token cap raised to 400.
2. **Emotion mapping fix** – curiosity is treated as a form of positive
   engagement → mapped to "Excited" for both Speaker A/B and reference labels.
3. **Prompt builders** – `build_generation_prompt()` and
   `build_baseline_prompt()` accept `target_words`.
4. **CLI flag** `--length-ratio` lets you change the multiplier (default 0.9).

Run example
-----------
```bash
python emotion_reply_pipeline_v2.py \
    --length-ratio 0.9 \
    --input path/to/pairs.json \
    --output-dir results/run2
``` 
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
from typing import Any, Dict, List, Tuple

from tqdm.auto import tqdm

# ------------------- Logging --------------------

def setup_logging(log_file: str | Path = "emotion_classification.log") -> logging.Logger:
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
OUTPUT_DIR = Path("/data/group_data/starlight/gpa/tts/multidialog_emotion_planning")
MAX_TOKENS_CLASS = 4
MAX_TOKENS_GEN = 400   # generous cap for long replies
TEMPERATURE_CLASS = 0.0
TEMPERATURE_GEN_DEFAULT = 0.7
MAX_WORKERS = 4
BATCH_SIZE = 8
RANDOM_SEED = 42
LENGTH_RATIO_DEFAULT = 0.9

ALLOWED_EMOTIONS: set[str] = {
    "Happy", "Sad", "Angry", "Neutral", "Surprise", "Disgust", "Fear", "Excited",
}

EMOTION_MAPPING: dict[str, str] = {
    "neutral": "Neutral",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "surprise": "Surprise",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happy",
    "excited": "Excited",
    # --- curiosity family mapped to Excited (positive engagement) ---
    "curious": "Excited",
    "curious to dive deeper": "Excited",
}

REFERENCE_EMOTION_MAPPING: dict[str, str] = {
    "surprised": "Surprise",
    "fearful": "Fear",
    "disgusted": "Disgust",
    "curious to dive deeper": "Excited",
}

# ------------------- Data Classes --------------------

@dataclass
class EntryResult:
    pair_id: str
    history: str
    speaker_name: str
    target_emotion: str
    emotion_steered_reply: str
    baseline_reply: str
    reference_emotion: str
    standardized_reference_emotion: str
    reference_response: str
    timestamp: str


# ------------------- Utility functions --------------------

def map_emotion(raw: str) -> str:
    key = raw.strip().lower()
    return EMOTION_MAPPING.get(key, raw.strip().title())


def standardize_reference_emotion(raw: str) -> str:
    key = raw.strip().lower()
    if key in REFERENCE_EMOTION_MAPPING:
        return REFERENCE_EMOTION_MAPPING[key]
    if key in EMOTION_MAPPING:
        return EMOTION_MAPPING[key]
    for allowed in ALLOWED_EMOTIONS:
        if allowed.lower() == key:
            return allowed
    return "Neutral"


# ------------------- Prompt builders --------------------

def build_classification_prompt(text: str, emotion_A: str) -> List[Dict[str, str]]:
    """
    Build a prompt to classify the emotional response to a given text.
    
    Args:
        text: The input text to classify
        emotion_A: The emotion of Speaker A
        
    Returns:
        A list of dictionaries representing the prompt for classification
    """
    prompt = [
        {"role": "system", "content": "You are an emotion classification system. Based on Speaker A's text, predict the most natural emotional response for Speaker B. Choose from: Happy, Sad, Angry, Neutral, Surprise, Disgust, Fear, Excited."},
        {"role": "user", "content": f"Speaker A ({emotion_A}): {text}\n\nWhat emotion would be the most natural for Speaker B to express in response? Answer with a single word."}
    ]
    return prompt

LONG_STYLE_GUIDE_TEMPLATE = (
    "Reply in 3–5 sentences totalling roughly **{target_words} words (±20)**. "
    "Your response should…\n"
    "• clearly embody the target EMOTION;\n"
    "• reference specific details from the last utterance;\n"
    "• add a personal, empathetic perspective;\n"
    "• end with a gentle question to keep the dialogue alive."
)

# (few‑shot examples unchanged from v2.0 for brevity)
from textwrap import dedent
FEWSHOT_LONG: list[tuple[str, str]] = [
    # ---------- SURPRISE ----------
    (
        "I didn't expect you to come!\nEMOTION=Surprise",
        dedent(
            """Whoa, I totally didn't see that coming! It's *such* a treat to bump into you like this. \
            I'm honestly smiling ear‑to‑ear right now—days like these remind me how fun life is when it throws a curveball. \
            Care to grab a coffee and tell me what inspired this nice surprise?"""
        ).strip(),
    ),
    # ---------- FEAR ----------
    (
        "The lights just went out…\nEMOTION=Fear",
        dedent(
            """Hey, it's okay—I'm right here with you. Take a slow breath; the power outage will probably be fixed in a minute or two. \
            Back when we were kids, my brother and I would tell silly stories in the dark to stay calm, and it always helped. \
            Would you like me to stay on the line and keep chatting until the lights are back?"""
        ).strip(),
    ),
    # ---------- HAPPY ----------
    (
        "I got the job offer today!\nEMOTION=Happy",
        dedent(
            """That is *phenomenal* news—congratulations! You worked so hard for this, and I could hear your excitement building each time we talked about the interviews. \
            I remember you saying the team felt like a perfect cultural fit; it's wonderful when the universe aligns like that. \
            How are you planning to celebrate this big win tonight?"""
        ).strip(),
    ),
    # ---------- SAD ----------
    (
        "I failed the exam again.\nEMOTION=Sad",
        dedent(
            """I'm really sorry, that must feel discouraging. I know how much effort you poured into studying—this outcome doesn't erase your dedication or your worth. \
            When I hit a similar wall last semester, taking a day to process and then revisiting the material with a study partner made a big difference. \
            Would talking through what tripped you up help right now?"""
        ).strip(),
    ),
    # ---------- ANGRY ----------
    (
        "You broke my phone.\nEMOTION=Angry",
        dedent(
            """Seriously? My phone was brand‑new and it wasn't cheap. I trusted you to be careful, and now I'm stuck without my main lifeline. \
            This isn't just an inconvenience—it's a real cost I can't ignore. \
            What do you propose we do to make this right?"""
        ).strip(),
    ),
    # ---------- DISGUST ----------
    (
        "This smells awful.\nEMOTION=Disgust",
        dedent(
            """Ugh, that stench is unbearable; it's turning my stomach. I can practically *taste* how rotten it is. \
            Let's seal it up and get some fresh air in here before it permeates everything. \
            Could you help me find a bag so we can toss it out quickly?"""
        ).strip(),
    ),
    # ---------- EXCITED ----------
    (
        "We can finally travel next month!\nEMOTION=Excited",
        dedent(
            """Yes! I've been day‑dreaming about this trip for ages and now it's really happening. Imagining us strolling unfamiliar streets and sampling local food gives me butterflies. \
            I'll pull out that shared checklist we drafted last spring so we can lock down the last details. \
            Which city on the itinerary are you most pumped about?"""
        ).strip(),
    ),
    # ---------- NEUTRAL (original) ----------
    (
        "I'll call you later about the plan.\nEMOTION=Neutral",
        "Sounds good, I'll keep my phone handy. Once we connect, we can walk through the timeline step by step and make sure we're on the same page. Talk to you soon—let me know if anything changes in the meantime?",
    )
]


def build_generation_prompt(history: str, emotion: str, target_words: int) -> List[Dict[str, str]]:
    guide = LONG_STYLE_GUIDE_TEMPLATE.format(target_words=target_words)
    shots: list[dict[str, str]] = [
        {"role": "system", "content": f"You are Speaker B. Target emotion = **{emotion}**. {guide}"},
    ]
    for hist, reply in FEWSHOT_LONG:
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})
    shots.append({"role": "user", "content": f"{history}\nEMOTION={emotion}"})
    return shots

BASELINE_GUIDE_TEMPLATE = (
    "Reply in 3–5 sentences totalling roughly **{target_words} words (±20)** in a *Neutral* tone. "
    "Be polite, factual, and keep the conversation moving with a simple follow‑up question at the end."
)

NEUTRAL_FEWSHOT = [
    ("I'll be at the library tonight.", "Okay, I'll swing by around seven. Good luck with the reading—do you need me to bring any snacks?"),
    ("Can you send me the file?", "Sure, I'll email it over within the next ten minutes. Let me know if the attachment doesn't come through, alright?"),
    ("The meeting starts at nine.", "Got it, thanks for the heads‑up. I'll log in a few minutes early to ensure the slides load properly—sound good?"),
    ("It's raining again.", "Yeah, the forecast was spot‑on. I'm grabbing my waterproof jacket before heading out—are you set for the wet weather?"),
    ("I'm cooking pasta for dinner.", "Nice, pasta sounds perfect. If you send me the recipe, I can prep the salad—does that work for you?"),
]


def build_baseline_prompt(history: str, target_words: int):
    guide = BASELINE_GUIDE_TEMPLATE.format(target_words=target_words)
    shots = [{"role": "system", "content": guide}]
    for hist, reply in NEUTRAL_FEWSHOT:
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})
    shots.append({"role": "user", "content": history})
    return shots


# ------------------- LLM helpers --------------------

def batch_generate(router, alias: str, prompts: List[Tuple[str, List[Dict[str, str]]]], max_tokens: int, temperature: float, batch_size=4, max_workers=4, desc="Generating"):
    results: dict[str, str] = {}
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    with tqdm(total=len(prompts), desc=desc) as pbar:
        for batch in batches:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(generate_single_response, router, alias, prompt, max_tokens, temperature): pid
                    for pid, prompt in batch
                }
                for fut in as_completed(futures):
                    pid = futures[fut]
                    try:
                        results[pid] = fut.result()
                    except Exception as e:
                        results[pid] = f"ERROR: {e}"
                    pbar.update(1)
    return results


def generate_single_response(router, alias: str, prompt, max_tokens: int, temperature: float):
    resp = router.inference_call(alias, prompt, max_tokens=max_tokens, temperature=temperature).strip()
    return resp.splitlines()[0] if resp.splitlines() else resp


# ------------------- Core pipeline --------------------

def generate_responses_batch(router, alias: str, entries: List[Dict[str, Any]], logger: logging.Logger, gen_temperature: float, length_ratio: float):
    classification_prompts = []
    for entry in entries:
        pid = entry["pair_id"]
        emo_a = map_emotion(entry.get("emotion_A", "")) or "Neutral"
        classification_prompts.append((f"{pid}_class", build_classification_prompt(entry["text_A"], emo_a)))

    logger.info(f"Predicting emotions for {len(classification_prompts)} pairs")
    class_results = batch_generate(router, alias, classification_prompts, MAX_TOKENS_CLASS, TEMPERATURE_CLASS, BATCH_SIZE, MAX_WORKERS, desc="Classifying")

    steered_prompts, baseline_prompts = [], []
    for entry in entries:
        pid = entry["pair_id"]
        target_raw = class_results.get(f"{pid}_class", "Neutral")
        target = map_emotion(target_raw) if target_raw in ALLOWED_EMOTIONS else "Neutral"

        words_in_A = len(entry["text_A"].split())
        target_words = max(60, int(words_in_A * length_ratio))

        steered_prompts.append((f"{pid}_steered", build_generation_prompt(entry["text_A"], target, target_words)))
        baseline_prompts.append((f"{pid}_baseline", build_baseline_prompt(entry["text_A"], target_words)))

    logger.info(f"Generating {len(steered_prompts)} steered replies")
    steered_results = batch_generate(router, alias, steered_prompts, MAX_TOKENS_GEN, gen_temperature, BATCH_SIZE, MAX_WORKERS, desc="Steered")

    logger.info(f"Generating {len(baseline_prompts)} baseline replies")
    baseline_results = batch_generate(router, alias, baseline_prompts, MAX_TOKENS_GEN, gen_temperature, BATCH_SIZE, MAX_WORKERS, desc="Baseline")

    results = []
    for entry in entries:
        pid = entry["pair_id"]
        target = map_emotion(class_results.get(f"{pid}_class", "Neutral"))
        ref_emo = map_emotion(entry.get("emotion_B", ""))
        std_ref = standardize_reference_emotion(entry.get("emotion_B", ""))
        results.append(vars(EntryResult(
            pair_id=pid,
            history=entry["text_A"],
            speaker_name=entry["speaker_B"],
            target_emotion=target,
            emotion_steered_reply=steered_results.get(f"{pid}_steered", "(no response)"),
            baseline_reply=baseline_results.get(f"{pid}_baseline", "(no response)"),
            reference_emotion=ref_emo,
            standardized_reference_emotion=std_ref,
            reference_response=entry["text_B"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )))
    return results

# ------------------- Dataset helpers --------------------

def load_pairs(fp: Path):
    try:
        return json.loads(fp.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed loading {fp}: {exc}")


def remap_and_downsample(pairs):
    for p in pairs:
        p["reference_emotion"] = map_emotion(p.get("emotion_B", ""))
        p["standardized_reference_emotion"] = standardize_reference_emotion(p.get("emotion_B", ""))
    freqs = Counter(p["standardized_reference_emotion"] for p in pairs)
    excited_cnt = freqs.get("Excited", 0)
    max_other = max(cnt for emo, cnt in freqs.items() if emo != "Excited") if freqs else 0
    if excited_cnt > max_other:
        random.seed(RANDOM_SEED)
        excited = [p for p in pairs if p["standardized_reference_emotion"] == "Excited"]
        others = [p for p in pairs if p["standardized_reference_emotion"] != "Excited"]
        pairs = others + random.sample(excited, max_other)
        random.shuffle(pairs)
    return pairs


def get_model_configs():
    return {
        "llama_3_70b_q4": {"model_id": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", "base_url": "http://babel-0-23:8081/v1", "is_chat": False, "template_builder": None},
        "llama_3_3b_q4": {"model_id": "AMead10/Llama-3.2-3B-Instruct-AWQ", "base_url": "http://babel-0-23:8083/v1", "is_chat": False, "template_builder": None},
        "mistral_7b_q4": {"model_id": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ", "base_url": "http://babel-0-23:8082/v1", "is_chat": False, "template_builder": None},
    }


def create_inference_client():
    from src_new.vllm_router import RawTemplateBuilder, VLLMChatCompletion
    cfgs = get_model_configs()
    for cfg in cfgs.values():
        cfg["template_builder"] = RawTemplateBuilder()
    return VLLMChatCompletion(cfgs)


def filter_top_n_by_length(pairs, n: int):
    return sorted(pairs, key=lambda p: len(p.get("text_A", "")), reverse=True)[:n]


# ------------------- Main --------------------

def main():
    p = argparse.ArgumentParser(description="Emotion‑aware reply generator (dynamic length)")
    p.add_argument("--input", "-i", default=str(PAIRS_METADATA))
    p.add_argument("--output-dir", "-o", default=str(OUTPUT_DIR))
    p.add_argument("--model", "-m", default="llama_3_70b_q4")
    p.add_argument("--workers", "-w", type=int, default=MAX_WORKERS)
    p.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE)
    p.add_argument("--filter-top", "-f", type=int, default=0)
    p.add_argument("--gen-temperature", type=float, default=TEMPERATURE_GEN_DEFAULT)
    p.add_argument("--length-ratio", type=float, default=LENGTH_RATIO_DEFAULT, help="Reply length = ratio × len(text_A) in words")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "emotion_reply.log")
    logger.info("Args: %s", vars(args))

    pairs = load_pairs(Path(args.input))
    if args.filter_top > 0:
        pairs = filter_top_n_by_length(pairs, args.filter_top)
    pairs = remap_and_downsample(pairs)
    logger.info("Dataset after preprocessing: %d pairs", len(pairs))

    client = create_inference_client()
    results, batch_size = [], args.batch_size
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        logger.info("Batch %d/%d", i // batch_size + 1, (len(pairs)+batch_size-1)//batch_size)
        results.extend(generate_responses_batch(client, args.model, batch, logger, args.gen_temperature, args.length_ratio))
        if (i // batch_size) % 5 == 0:
            tmp = out_dir / f"partial_{len(results)}.jsonl"; tmp.write_text("\n".join(json.dumps(r) for r in results))
    (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results))

    stats = {
        "total": len(results),
        "emotion_distribution": Counter(r["target_emotion"] for r in results),
        "agreement": sum(1 for r in results if r["target_emotion"] == r["standardized_reference_emotion"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    stats["agreement_pct"] = 100 * stats["agreement"] / stats["total"] if stats["total"] else 0
    (out_dir / "stats.json").write_text(json.dumps({k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}, indent=2))
    logger.info("Done – processed %d pairs", len(results))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        traceback.print_exc(); sys.exit(1)
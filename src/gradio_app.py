#!/usr/bin/env python3
"""gradio_app_multi_model.py
A Gradio demo allowing selection of multiple ASR and dummy TTS backends,
plus a gold-standard Whisper transcript and key ASR metrics:
  • Choose between multiple ESPnet ASR models
  • Choose between multiple dummy TTS strategies
  • Record up to ~10 s of speech → ASR → transcript
  • Infuse with a selected emotion tag
  • Display latency, real‑time‑factor, WER & CER
  • Show Whisper‑large “gold” transcript

Run:
    # install dependencies first:
    pip install openai-whisper jiwer gradio librosa soundfile torch espnet_model_zoo espnet
    python gradio_app_multi_model.py
"""

import logging
import string
import time
import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Callable

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import whisper
from jiwer import wer, cer
from espnet2.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = Path("asr_debug.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("asr_multi")
logger.info("Starting multi-model ASR/TTS demo")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", DEVICE)

# ---------------------------------------------------------------------------
# ASR model tags
# ---------------------------------------------------------------------------
ASR_MODEL_TAGS: Dict[str, str] = {
    "Conformer6": (
        "kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_"
        "hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave"
    ),
    "Conformer8": "pyf98/librispeech_conformer",
}

# ---------------------------------------------------------------------------
# Dummy TTS functions
# ---------------------------------------------------------------------------
def silence_tts(text: str, emotion: str, sample_rate: int) -> Tuple[int, np.ndarray]:
    duration = 1.0
    return sample_rate, np.zeros(int(duration * sample_rate), dtype=np.float32)

def tone_tts(text: str, emotion: str, sample_rate: int) -> Tuple[int, np.ndarray]:
    duration = 1.0
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    wav = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return sample_rate, wav

DUMMY_TTS_MODELS: Dict[str, Callable[[str, str, int], Tuple[int, np.ndarray]]] = {
    "Silence": silence_tts,
    "Tone": tone_tts,
}

# ---------------------------------------------------------------------------
# Load ESPnet ASR models
# ---------------------------------------------------------------------------
downloader = ModelDownloader()
asr_models: Dict[str, Speech2Text] = {}

for name, tag in ASR_MODEL_TAGS.items():
    logger.info("Downloading ASR model '%s' → %s", name, tag)
    model_files = downloader.download_and_unpack(tag)

    asr_kwargs = {
        "device": DEVICE,
        "beam_size": 10,
        "ctc_weight": 0.3,
        "maxlenratio": 0.0,
        "minlenratio": 0.0,
        "nbest": 1,
    }
    if "bpemodel" in model_files:
        asr_kwargs["token_type"] = "bpe"
        bpemodel = model_files["bpemodel"]
        if os.path.exists(bpemodel):
            asr_kwargs["bpemodel"] = bpemodel
        else:
            parent = Path(list(model_files.values())[0]).parent.parent
            alt = list(parent.glob("**/*model.model"))
            if alt:
                model_files["bpemodel"] = str(alt[0])
                asr_kwargs["bpemodel"] = str(alt[0])
            else:
                logger.warning("Skipping %s: no BPE model found", name)
                continue
    try:
        asr_models[name] = Speech2Text(**model_files, **asr_kwargs)
        logger.info("Loaded ESPnet ASR model: %s", name)
    except Exception as e:
        logger.error("Failed to load %s: %s", name, e)

if not asr_models:
    logger.error("No ASR models loaded; exiting.")
    exit(1)

MODEL_SR = 16000

# ---------------------------------------------------------------------------
# Load Whisper “gold” model
# ---------------------------------------------------------------------------
logger.info("Loading Whisper large as gold ASR…")
whisper_model = whisper.load_model("large")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMOTIONS = ["neutral","happy","sad","angry","surprise","fear","disgust","excited"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_audio_sample(wav: np.ndarray, sr: int, prefix: str) -> None:
    out = Path("debug_samples")
    out.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = out / f"{prefix}_{ts}.wav"
    sf.write(fn, wav, sr)
    logger.info("Saved sample %s (%.2f s)", fn, len(wav)/sr)

def text_normalizer(text: str) -> str:
    text = text.replace("▁", " ")
    text = " ".join(text.split())
    return text.upper().translate(str.maketrans("", "", string.punctuation))

# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------
def transcribe_and_speak(
    asr_choice: str,
    tts_choice: str,
    input_wav: Optional[Tuple[int, np.ndarray]],
    emotion: str
) -> Tuple[str, str, str, str, Tuple[int, np.ndarray]]:
    """ Returns: (hypothesis, emotion, metrics_str, gold_whisper, tts_audio) """
    if not input_wav:
        empty = (MODEL_SR, np.zeros(MODEL_SR, dtype=np.float32))
        return "No audio provided.", emotion, "-", "", empty

    # Prepare audio
    user_sr, wav = input_wav
    wav = wav.astype(np.float32)
    if len(wav)/user_sr > 12:
        empty = (MODEL_SR, np.zeros(MODEL_SR, dtype=np.float32))
        return "Please speak ≤10 s.", emotion, "Too long", "", empty
    peak = np.max(np.abs(wav))
    if peak > 1.0:
        wav /= peak
    if np.max(np.abs(wav)) < 0.01:
        empty = (MODEL_SR, np.zeros(MODEL_SR, dtype=np.float32))
        return "Audio too quiet.", emotion, "Low volume", "", empty
    if user_sr != MODEL_SR:
        wav = librosa.resample(wav, orig_sr=user_sr, target_sr=MODEL_SR)
        user_sr = MODEL_SR
    save_audio_sample(wav, user_sr, "input")

    # ESPnet ASR → hypothesis
    model = asr_models.get(asr_choice)
    start = time.time()
    try:
        nbests = model(wav)
    except Exception as e:
        empty = (MODEL_SR, np.zeros(MODEL_SR, dtype=np.float32))
        return f"ASR ERROR: {e}", emotion, "Fail", "", empty
    latency = time.time() - start

    if not nbests:
        empty = (MODEL_SR, np.zeros(MODEL_SR, dtype=np.float32))
        return "(No hypothesis)", emotion, "Empty", "", empty

    # Unpack and normalize hypothesis
    hyp_text = nbests[0][0]
    hyp_text = hyp_text.replace("▁", " ").strip()
    hyp = text_normalizer(hyp_text)

    # Whisper gold transcript
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, MODEL_SR)
        gold = whisper_model.transcribe(tmp.name)["text"].strip()

    # WER & CER
    wer_err = wer(gold.upper(), hyp)
    cer_err = cer(gold.upper(), hyp)

    # RTF
    rtf = latency / (len(wav)/MODEL_SR) if len(wav) > 0 else 0.0

    # Build metrics string
    metrics_parts = [
        f"Latency: {latency:.2f}s",
        f"RTF: {rtf:.2f}",
        f"WER: {wer_err*100:.1f}%",
        f"CER: {cer_err*100:.1f}%"
    ]
    metrics_str = " | ".join(metrics_parts)

    # Dummy TTS
    tts_sr, tts_wav = DUMMY_TTS_MODELS[tts_choice](hyp, emotion, sample_rate=MODEL_SR)
    return hyp, emotion, metrics_str, gold, (tts_sr, tts_wav)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
inputs = [
    gr.Dropdown(choices=list(asr_models.keys()), label="ASR Model"),
    gr.Dropdown(choices=list(DUMMY_TTS_MODELS.keys()), label="Dummy TTS Model"),
    gr.Audio(source="microphone", type="numpy", label="Speak here"),
    gr.Dropdown(EMOTIONS, value="neutral", label="Infuse emotion"),
]
outputs = [
    gr.Textbox(label="Transcript"),
    gr.Textbox(label="Emotion"),
    gr.Textbox(label="Metrics"),
    gr.Textbox(label="Gold (Whisper)", interactive=False),
    gr.Audio(label="Dummy TTS Output"),
]

demo = gr.Interface(
    fn=transcribe_and_speak,
    inputs=inputs,
    outputs=outputs,
    title="Multi-Model ASR + Dummy TTS Demo",
    description="Select ASR & TTS backends, speak ≤10 s, and inspect results (incl. Whisper gold)."
)

if __name__ == "__main__":
    logger.info("Launching Gradio app…")
    demo.launch(share=True)

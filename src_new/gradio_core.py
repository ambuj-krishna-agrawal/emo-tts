#!/usr/bin/env python3
"""
Emotion-Aware Speech Processing Core Module
This module provides the core functionality for the emotion-aware speech processing system,
including ASR, emotion detection, response generation, and TTS.
"""

import logging
import time
import tempfile
from pathlib import Path
import os
import json
import re
import numpy as np
import soundfile as sf
import torch
import whisper

# Import our emotion detection and response module
from src_new.emotion_detection_and_response import (
    detect_emotion,
    generate_emotional_response,
    generate_neutral_response,
    ALLOWED_EMOTIONS
)

# Import EmotiVoice TTS function
from src_new.multidialogue_tts_batch_synth_emotivoice_2 import generate_emotivoice_audio

# ---------------------------------------------------------------------------
# Basic Setup
# ---------------------------------------------------------------------------
LOG_PATH = Path("emotion_asr_debug.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("emotion_asr")
logger.info("Starting emotion-aware ASR/TTS core module with advanced emotion detection, EmotiVoice TTS, and emotion2vec metrics")

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Constants
MODEL_SR = 16000
EMOTIONS = list(ALLOWED_EMOTIONS)
DEFAULT_MODEL = "llama_3_3b_q4"  # Use smaller model for faster responses in demo

# Emotion mapping for emotion2vec
EMOTION_MAPPING = {
    "angry": 0, "anger": 0, "frustrated": 0,
    "disgust": 1, "disgusted": 1,
    "fear": 2, "fearful": 2,
    "happy": 3, "joy": 3, "excited": 3,
    "neutral": 4,
    "other": 5, "none": 5,
    "sad": 6, "sadness": 6, "bored": 6,
    "surprise": 7, "surprised": 7, "curious": 7,
}

# ---------------------------------------------------------------------------
# Load Whisper model
# ---------------------------------------------------------------------------
logger.info("Loading Whisper model for ASR...")
whisper_model = whisper.load_model("large")

# ---------------------------------------------------------------------------
# Emotion2vec Analysis
# ---------------------------------------------------------------------------
def analyze_emotion_with_emotion2vec(audio_path):
    """
    Analyze the emotion in an audio file using emotion2vec
    Returns emotion prediction and probabilities
    """
    try:
        # Try to import FunASR
        try:
            from funasr import AutoModel
        except ImportError as exc:
            logger.error(f"FunASR missing: {exc}")
            return {
                "predicted_emotion_label": "unknown",
                "predicted_emotion_idx": -1,
                "predicted_emotion_score": 0.0,
                "scores": [],
                "labels": []
            }
        
        # Load the emotion2vec model
        try:
            model = AutoModel(model="iic/emotion2vec_plus_large")
        except Exception:
            logger.info("Falling back to base_finetuned model")
            model = AutoModel(model="iic/emotion2vec_base_finetuned")
        
        # Generate emotion prediction
        rec = model.generate(str(audio_path), granularity="utterance", extract_embedding=False)
        
        # Normalize output
        if isinstance(rec, list) and rec and isinstance(rec[0], dict):
            rec = rec[0]
        
        scores = rec.get("scores") or rec.get("score")
        labels = rec.get("labels") or rec.get("label")
        
        if scores is None or labels is None:
            logger.error(f"Bad emotion2vec output for {audio_path}")
            return {
                "predicted_emotion_label": "unknown",
                "predicted_emotion_idx": -1,
                "predicted_emotion_score": 0.0,
                "scores": [],
                "labels": []
            }
        
        idx = int(np.argmax(scores))
        
        return {
            "scores": scores,
            "labels": labels,
            "predicted_emotion_idx": idx,
            "predicted_emotion_score": float(scores[idx]),
            "predicted_emotion_label": labels[idx],
        }
    
    except Exception as e:
        logger.error(f"Error analyzing emotion with emotion2vec: {e}")
        return {
            "predicted_emotion_label": "unknown",
            "predicted_emotion_idx": -1,
            "predicted_emotion_score": 0.0,
            "scores": [],
            "labels": []
        }

_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

def _ascii_only(txt: str) -> str:
    """Strip every non‑ASCII character (keeps spaces, punctuation, digits)."""
    return _ASCII_RE.sub("", txt)

def get_emotion_metrics(audio_path, target_emotion=None):
    """
    Get real emotion metrics for the given audio using emotion2vec.
    Returns a single ASCII‑only, multi‑line string.
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file does not exist: {audio_path}")
        return "Error: Audio file not found"

    emo = analyze_emotion_with_emotion2vec(audio_path)

    # ------------------------------------------------------------------ #
    # Clean labels → ASCII only, keep original order
    # ------------------------------------------------------------------ #
    labels = [_ascii_only(l) for l in emo.get("labels", [])]
    scores = emo.get("scores", [])
    idx      = emo.get("predicted_emotion_idx", -1)
    conf     = emo.get("predicted_emotion_score", 0.0)
    predicted = labels[idx] if 0 <= idx < len(labels) else "unknown"

    # score lines "Happy: 0.82"
    score_lines = [
        f"{lab}: {scr:.2f}" for lab, scr in zip(labels, scores)
        if lab  # ignore empty labels after stripping
    ]
    score_block = "\n".join(score_lines)

    # optional match symbol
    match_info = ""
    if target_emotion:
        tgt_idx = EMOTION_MAPPING.get(target_emotion.lower(), -1)
        pred_idx = EMOTION_MAPPING.get(predicted.lower(), -2)
        if tgt_idx >= 0 and pred_idx >= 0:
            match_info = "✓" if tgt_idx == pred_idx else "✗"

    return (
        f"Predicted: {predicted}   Conf: {conf:.2f}   Match: {match_info}\n"
        f"{score_block}"
    )

# ---------------------------------------------------------------------------
# TTS Functions
# ---------------------------------------------------------------------------
def generate_dummy_speech(text, emotion, sr=MODEL_SR):
    """Generate a simple tone with characteristics based on emotion (fallback)"""
    duration = max(len(text.split()) * 0.3, 1.0)  # rough estimate: 0.3s per word, min 1s
    
    # Create simple waveform
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    
    # Different frequency for different emotions
    freq_map = {
        "Neutral": 220.0,
        "Happy": 293.66,
        "Sad": 196.0,
        "Angry": 329.63,
        "Surprise": 349.23,
        "Fear": 185.0,
        "Disgust": 174.61,
        "Excited": 392.0,
    }
    
    freq = freq_map.get(emotion, 220.0)
    amplitude = 0.2
    
    # Generate sine wave
    wav = amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    return sr, wav

def get_emotivoice_audio(text, emotion, output_dir=None):
    """Generate audio using EmotiVoice with emotion"""
    try:
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="emotivoice_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Call the EmotiVoice TTS function
        audio_path = generate_emotivoice_audio(
            text=text,
            target_emotion=emotion,
            output_dir=output_dir
        )
        
        # Read the generated audio file
        sr, wav = sf.read(str(audio_path))
        return sr, wav, str(audio_path)
    except Exception as e:
        logger.error(f"Error generating EmotiVoice audio: {e}")
        # Fall back to dummy speech if EmotiVoice fails
        logger.info("Falling back to dummy speech generation")
        sr, wav = generate_dummy_speech(text, emotion)
        
        # Save the dummy audio to a file
        fallback_path = output_dir / f"fallback_{emotion.lower()}.wav"
        sf.write(fallback_path, wav, sr)
        
        return sr, wav, str(fallback_path)

def get_audio_duration(audio_path):
    """Get the duration of an audio file in seconds"""
    try:
        file_info = sf.info(audio_path)
        return file_info.duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return 1.0  # Default to 1 second if there's an error

# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------
def process_audio(audio_path, tts_model="EmotiVoice", llm_model=DEFAULT_MODEL):
    """Main function to process audio from a file path"""
    if not audio_path:
        return {
            "transcript": "No audio provided",
            "asr_metrics": "",
            "detected_emotion": "",
            "emotional_response": "",
            "neutral_response": "",
            "target_emotion": "",
            "emotional_audio_path": None,
            "emotional_metrics": "",
            "emotional_neutral_audio_path": None,
            "emotional_neutral_metrics": "",
            "neutral_audio_path": None,
            "neutral_metrics": "",
            "tts_model": tts_model
        }
    
    # Load audio file
    try:
        sr, wav = sf.read(audio_path)
        audio_duration = get_audio_duration(audio_path)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return {
            "transcript": f"Error loading audio: {e}",
            "asr_metrics": "",
            "detected_emotion": "",
            "emotional_response": "",
            "neutral_response": "",
            "target_emotion": "",
            "emotional_audio_path": None,
            "emotional_metrics": "",
            "emotional_neutral_audio_path": None,
            "emotional_neutral_metrics": "",
            "neutral_audio_path": None,
            "neutral_metrics": "",
            "tts_model": tts_model
        }
    
    # Process with Whisper
    start_time = time.time()
    try:
        result = whisper_model.transcribe(audio_path, language='en')
        transcript = result["text"].strip()
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return {
            "transcript": f"Transcription error: {e}",
            "asr_metrics": "",
            "detected_emotion": "",
            "emotional_response": "",
            "neutral_response": "",
            "target_emotion": "",
            "emotional_audio_path": None,
            "emotional_metrics": "",
            "emotional_neutral_audio_path": None,
            "emotional_neutral_metrics": "",
            "neutral_audio_path": None,
            "neutral_metrics": "",
            "tts_model": tts_model
        }
    
    # Calculate ASR metrics
    latency = time.time() - start_time
    rtf = latency / audio_duration if audio_duration > 0 else 0
    asr_metrics = f"Latency: {latency:.2f}s | RTF: {rtf:.2f}"
    
    # Use our improved emotion detection
    try:
        # Default speaker emotion to Neutral
        target_emotion = detect_emotion(transcript, llm_model)
        logger.info(f"Detected emotion: {target_emotion}")
        
        # Generate emotion-steered response
        emotional_response = generate_emotional_response(transcript, target_emotion, 0.9, llm_model, 0.7)
        
        # Generate neutral response
        neutral_response = generate_neutral_response(transcript, 0.9, llm_model, 0.7)
    except Exception as e:
        logger.error(f"Error in emotion detection or response generation: {e}")
        target_emotion = "Neutral"
        emotional_response = f"Error generating response: {e}"
        neutral_response = f"Error generating response: {e}"
    
    # Create output directory for TTS
    output_dir = Path(tempfile.mkdtemp(prefix="emotion_tts_"))
    
    # Generate speech based on selected TTS model
    logger.info(f"Generating speech using {tts_model}")
    
    if tts_model == "EmotiVoice":
        try:
            # Generate emotional response with emotional delivery
            emotional_sr, emotional_wav, emotional_audio_path = get_emotivoice_audio(
                emotional_response, target_emotion, output_dir
            )
            
            # Generate emotional response with neutral delivery
            emotional_neutral_sr, emotional_neutral_wav, emotional_neutral_audio_path = get_emotivoice_audio(
                emotional_response, "Neutral", output_dir
            )
            
            # Generate neutral response with neutral delivery
            neutral_sr, neutral_wav, neutral_audio_path = get_emotivoice_audio(
                neutral_response, "Neutral", output_dir
            )
        except Exception as e:
            logger.error(f"EmotiVoice generation error: {e}")
            # Fall back to dummy TTS
            emotional_sr, emotional_wav = generate_dummy_speech(emotional_response, target_emotion)
            emotional_audio_path = os.path.join(tempfile.gettempdir(), f"fallback_emotional_response.wav")
            sf.write(emotional_audio_path, emotional_wav, emotional_sr)
            
            emotional_neutral_sr, emotional_neutral_wav = generate_dummy_speech(emotional_response, "Neutral")
            emotional_neutral_audio_path = os.path.join(tempfile.gettempdir(), f"fallback_emotional_neutral_response.wav")
            sf.write(emotional_neutral_audio_path, emotional_neutral_wav, emotional_neutral_sr)
            
            neutral_sr, neutral_wav = generate_dummy_speech(neutral_response, "Neutral")
            neutral_audio_path = os.path.join(tempfile.gettempdir(), f"fallback_neutral_response.wav")
            sf.write(neutral_audio_path, neutral_wav, neutral_sr)
    else:
        # Use dummy TTS for other models
        emotional_sr, emotional_wav = generate_dummy_speech(emotional_response, target_emotion)
        emotional_audio_path = os.path.join(tempfile.gettempdir(), f"{tts_model}_emotional_response.wav")
        sf.write(emotional_audio_path, emotional_wav, emotional_sr)
        
        emotional_neutral_sr, emotional_neutral_wav = generate_dummy_speech(emotional_response, "Neutral")
        emotional_neutral_audio_path = os.path.join(tempfile.gettempdir(), f"{tts_model}_emotional_neutral_response.wav")
        sf.write(emotional_neutral_audio_path, emotional_neutral_wav, emotional_neutral_sr)
        
        neutral_sr, neutral_wav = generate_dummy_speech(neutral_response, "Neutral")
        neutral_audio_path = os.path.join(tempfile.gettempdir(), f"{tts_model}_neutral_response.wav")
        sf.write(neutral_audio_path, neutral_wav, neutral_sr)
    
    # Get emotion metrics using emotion2vec
    emotional_metrics_str = get_emotion_metrics(emotional_audio_path, target_emotion)
    emotional_neutral_metrics_str = get_emotion_metrics(emotional_neutral_audio_path, "Neutral")
    neutral_metrics_str = get_emotion_metrics(neutral_audio_path, "Neutral")
    
    return {
        "transcript": transcript,
        "asr_metrics": asr_metrics,
        "detected_emotion": target_emotion,
        "emotional_response": emotional_response,
        "neutral_response": neutral_response,
        "target_emotion": target_emotion,
        "emotional_audio_path": emotional_audio_path,
        "emotional_metrics": emotional_metrics_str,
        "emotional_neutral_audio_path": emotional_neutral_audio_path,
        "emotional_neutral_metrics": emotional_neutral_metrics_str,
        "neutral_audio_path": neutral_audio_path,
        "neutral_metrics": neutral_metrics_str,
        "tts_model": tts_model
    }

# Function to process text directly (no audio)
def process_text(text, llm_model=DEFAULT_MODEL):
    """Process text input directly without audio"""
    if not text:
        return {
            "detected_emotion": "",
            "emotional_response": "",
            "neutral_response": "",
        }
    
    try:
        # Detect emotion
        target_emotion = detect_emotion(text, llm_model)
        
        # Generate responses
        emotional_response = generate_emotional_response(text, target_emotion, 0.9, llm_model)
        neutral_response = generate_neutral_response(text, 0.9, llm_model)
        
        return {
            "detected_emotion": target_emotion,
            "emotional_response": emotional_response,
            "neutral_response": neutral_response,
        }
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return {
            "detected_emotion": "Error",
            "emotional_response": f"Error: {e}",
            "neutral_response": f"Error: {e}",
        }


def analyze_embedding_similarity(gt_audio_path: str, gen_audio_path: str):
    """
    Compare the emotion embeddings of a ground-truth audio vs. a generated audio.
    Returns a dict with:
      - gt_embedding: list of floats
      - gen_embedding: list of floats
      - cosine_similarity: float in [–1,1]
    """
    try:
        from funasr import AutoModel
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        logger.error(f"FunASR not installed for embedding-based eval: {exc}")
        return {"gt_embedding": None, "gen_embedding": None, "cosine_similarity": None}

    # load the same emotion2vec model (or fallback)
    try:
        embed_model = AutoModel(model="iic/emotion2vec_plus_large")
    except Exception:
        logger.info("Falling back to base_finetuned emotion2vec for embeddings")
        embed_model = AutoModel(model="iic/emotion2vec_base_finetuned")

    # helper to pull out a single utterance embedding
    def _get_embedding(path):
        rec = embed_model.generate(
            str(path),
            granularity="utterance",
            extract_embedding=True
        )
        # Some versions return a list of dicts; grab the first
        if isinstance(rec, list) and rec:
            rec = rec[0]
        feats = rec.get("feats") or rec.get("embedding")
        if feats is None:
            raise RuntimeError(f"No embedding found in model output for {path}")
        return torch.from_numpy(feats)

    try:
        gt_emb = _get_embedding(gt_audio_path)
        gen_emb = _get_embedding(gen_audio_path)
        # cosine over the vector dims → single score
        cos = F.cosine_similarity(gt_emb, gen_emb, dim=0).mean().item()
    except Exception as e:
        logger.error(f"Error computing embeddings for {gt_audio_path} vs. {gen_audio_path}: {e}")
        return {"gt_embedding": None, "gen_embedding": None, "cosine_similarity": None}

    return {
        "gt_embedding": gt_emb.numpy().tolist(),
        "gen_embedding": gen_emb.numpy().tolist(),
        "cosine_similarity": cos
    }

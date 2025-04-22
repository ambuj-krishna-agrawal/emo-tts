#!/usr/bin/env python3
"""
Emotion-Aware Speech Processing Gradio App
This version integrates the emotion_detection_and_response module
with the Whisper ASR and EmotiVoice TTS components, and adds real
emotion metrics using emotion2vec
"""

import logging
import time
import tempfile
from pathlib import Path
import os
import json

# Specify gradio version to avoid incompatibilities
os.environ["GRADIO_SERVER_PORT"] = "7860"

# Import after setting environment variables
import numpy as np
import soundfile as sf
import torch
import whisper

# Import our new emotion detection and response module
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
logger.info("Starting emotion-aware ASR/TTS demo with advanced emotion detection, EmotiVoice TTS, and emotion2vec metrics")

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
    
import re

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

    # score lines “Happy: 0.82”
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
            # Generate emotional response audio
            emotional_sr, emotional_wav, emotional_audio_path = get_emotivoice_audio(
                emotional_response, target_emotion, output_dir
            )
            
            # Generate neutral response audio
            neutral_sr, neutral_wav, neutral_audio_path = get_emotivoice_audio(
                neutral_response, "Neutral", output_dir
            )
        except Exception as e:
            logger.error(f"EmotiVoice generation error: {e}")
            # Fall back to dummy TTS
            emotional_sr, emotional_wav = generate_dummy_speech(emotional_response, target_emotion)
            emotional_audio_path = os.path.join(tempfile.gettempdir(), f"fallback_emotional_response.wav")
            sf.write(emotional_audio_path, emotional_wav, emotional_sr)
            
            neutral_sr, neutral_wav = generate_dummy_speech(neutral_response, "Neutral")
            neutral_audio_path = os.path.join(tempfile.gettempdir(), f"fallback_neutral_response.wav")
            sf.write(neutral_audio_path, neutral_wav, neutral_sr)
    else:
        # Use dummy TTS for other models
        emotional_sr, emotional_wav = generate_dummy_speech(emotional_response, target_emotion)
        emotional_audio_path = os.path.join(tempfile.gettempdir(), f"{tts_model}_emotional_response.wav")
        sf.write(emotional_audio_path, emotional_wav, emotional_sr)
        
        neutral_sr, neutral_wav = generate_dummy_speech(neutral_response, "Neutral")
        neutral_audio_path = os.path.join(tempfile.gettempdir(), f"{tts_model}_neutral_response.wav")
        sf.write(neutral_audio_path, neutral_wav, neutral_sr)
    
    # Get emotion metrics using emotion2vec
    emotional_metrics_str = get_emotion_metrics(emotional_audio_path, target_emotion)
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

# ---------------------------------------------------------------------------
# Gradio interface with side-by-side layout and text input option
# ---------------------------------------------------------------------------
def run_gradio():
    """Run the Gradio interface with side-by-side layout"""
    import gradio as gr
    
    # Gradio wrapper for audio processing
    def gradio_audio_wrapper(audio, tts_model_choice, llm_model_choice):
        if audio is None:
            return "No audio provided", "", "", "", "", "", "", None, "", None, ""
        
        # Save the uploaded audio to a temporary file
        audio_path = os.path.join(tempfile.gettempdir(), "input.wav")
        
        try:
            sr, data = audio
            sf.write(audio_path, data, sr)
        except Exception as e:
            logger.error(f"Error processing audio input: {e}")
            return f"Error processing audio: {e}", "", "", "", "", "", "", None, "", None, ""
        
        # Process the audio with the selected models
        result = process_audio(audio_path, tts_model_choice, llm_model_choice)
        
        # Return results in the format Gradio expects
        return (
            result["transcript"],
            result["asr_metrics"],
            result["detected_emotion"],
            result["emotional_response"],
            result["neutral_response"],
            result["target_emotion"],
            result["tts_model"],
            result["emotional_audio_path"],
            result["emotional_metrics"],
            result["neutral_audio_path"],
            result["neutral_metrics"]
        )
    
    # Gradio wrapper for text processing
    def gradio_text_wrapper(text, llm_model_choice):
        if not text:
            return "", "", ""
        
        # Process the text
        result = process_text(text, llm_model_choice)
        
        return (
            result["detected_emotion"],
            result["emotional_response"],
            result["neutral_response"]
        )
    
    # Get available LLM models 
    available_models = ["llama_3_70b_q4", "llama_3_3b_q4", "mistral_7b_q4"]
    
    # Create a tabbed interface
    with gr.Blocks(title="Emotion-Aware Speech & Text Demo") as demo:
        gr.Markdown("# Emotion-Aware Speech & Text Processing Demo")
        gr.Markdown("This demo showcases emotion detection and emotional response generation with emotion2vec analysis")
        
        with gr.Tabs():
            # Audio Processing Tab
            with gr.TabItem("Speech Processing"):
                gr.Markdown("## Speech Processing")
                gr.Markdown("Record your voice or upload audio to see how emotion is detected and used for response generation")
                
                with gr.Row():
                    # Left column - Input
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        audio_input = gr.Audio(label="Speak here", type="numpy")
                        
                        with gr.Row():
                            # TTS model selection
                            tts_model_choice = gr.Radio(
                                ["EmotiVoice", "ChatGPT"], 
                                label="Select TTS Model", 
                                value="EmotiVoice"
                            )
                            
                            # LLM model selection
                            llm_model_choice_audio = gr.Dropdown(
                                available_models,
                                label="Select LLM Model", 
                                value=DEFAULT_MODEL
                            )
                        
                        submit_audio_btn = gr.Button("Process Speech", variant="primary")
                    
                    # Right column - Results
                    with gr.Column(scale=1):
                        gr.Markdown("### Results")
                        # ASR Results section
                        with gr.Group():
                            gr.Markdown("#### ASR Results")
                            transcript_output = gr.Textbox(label="Transcript")
                            asr_metrics_output = gr.Textbox(label="ASR Metrics")
                            detected_emotion_output = gr.Textbox(label="Detected Emotion")
                        
                        # Response section
                        with gr.Group():
                            gr.Markdown("#### Generated Responses")
                            
                            # Emotion-steered response
                            with gr.Group():
                                gr.Markdown("##### Emotion-Steered Response")
                                emotional_response_output = gr.Textbox(label="Response Text")
                                target_emotion_output = gr.Textbox(label="Target Emotion")
                                tts_model_output = gr.Textbox(label="TTS Model Used")
                            
                            # Neutral response
                            with gr.Group():
                                gr.Markdown("##### Neutral Response")
                                neutral_response_output = gr.Textbox(label="Response Text")
                        
                        # Audio Responses section
                        with gr.Group():
                            gr.Markdown("#### Audio Responses")
                            # Emotional response
                            with gr.Group():
                                gr.Markdown("##### Emotion-Steered Audio")
                                emotional_audio_output = gr.Audio(label="Audio")
                                emotional_metrics_output = gr.Textbox(label="Emotion2vec Analysis", lines=4, max_lines=10)
                            
                            # Neutral response
                            with gr.Group():
                                gr.Markdown("##### Neutral Audio")
                                neutral_audio_output = gr.Audio(label="Audio")
                                neutral_metrics_output = gr.Textbox(label="Emotion2vec Analysis", lines=4, max_lines=10)
            
            # Text Processing Tab
            with gr.TabItem("Text Processing"):
                gr.Markdown("## Text Processing")
                gr.Markdown("Enter text to detect emotion and generate appropriate responses")
                
                with gr.Row():
                    # Left column - Input
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        text_input = gr.Textbox(
                            label="Enter text here", 
                            placeholder="Type your message...",
                            lines=5
                        )
                        
                        # LLM model selection
                        llm_model_choice_text = gr.Dropdown(
                            available_models,
                            label="Select LLM Model", 
                            value=DEFAULT_MODEL
                        )
                        
                        submit_text_btn = gr.Button("Process Text", variant="primary")
                    
                    # Right column - Results
                    with gr.Column(scale=1):
                        gr.Markdown("### Results")
                        
                        # Emotion detection result
                        text_emotion_output = gr.Textbox(label="Detected Emotion")
                        
                        # Response section
                        with gr.Group():
                            gr.Markdown("#### Generated Responses")
                            
                            # Emotion-steered response
                            with gr.Group():
                                gr.Markdown("##### Emotion-Steered Response")
                                text_emotional_response = gr.Textbox(label="Response", lines=5)
                            
                            # Neutral response
                            with gr.Group():
                                gr.Markdown("##### Neutral Response")
                                text_neutral_response = gr.Textbox(label="Response", lines=5)
        
        # Connect the buttons
        submit_audio_btn.click(
            fn=gradio_audio_wrapper,
            inputs=[audio_input, tts_model_choice, llm_model_choice_audio],
            outputs=[
                transcript_output,
                asr_metrics_output,
                detected_emotion_output,
                emotional_response_output,
                neutral_response_output,
                target_emotion_output,
                tts_model_output,
                emotional_audio_output,
                emotional_metrics_output,
                neutral_audio_output,
                neutral_metrics_output
            ]
        )
        
        submit_text_btn.click(
            fn=gradio_text_wrapper,
            inputs=[text_input, llm_model_choice_text],
            outputs=[
                text_emotion_output,
                text_emotional_response,
                text_neutral_response
            ]
        )
    
    # Launch the app
    logger.info("Launching Gradio interface...")
    demo.launch(share=True)
    return True

if __name__ == "__main__":
    logger.info("Starting application with Gradio interface...")
    print("Launching Gradio interface...")
    run_gradio()
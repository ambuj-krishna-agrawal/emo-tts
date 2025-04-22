#!/usr/bin/env python3
"""
Gradio Interface for Emotion‑Aware Speech Processing (fixed)
This version delivers exactly three audio clips:
  1. emotion‑steered response spoken with the target emotion
  2. emotion‑steered response spoken neutrally
  3. neutral response spoken neutrally

Only the first two clips are analysed with emotion2vec.
"""

import os
import tempfile
import logging
from pathlib import Path
import soundfile as sf

# ---------------------------------------------------------------------------
# Environment & libs
# ---------------------------------------------------------------------------
os.environ.setdefault("GRADIO_SERVER_PORT", "7860")

import gradio as gr  # type: ignore

from src_new.gradio_core import (
    process_audio,
    process_text,
    DEFAULT_MODEL,
    logger,
)

# ---------------------------------------------------------------------------
# Gradio helpers
# ---------------------------------------------------------------------------

def gradio_audio_wrapper(audio, tts_model_choice, llm_model_choice):
    """Handle speech input → run pipeline → return values for UI."""
    if audio is None:
        return (
            "No audio provided", "", "", "", "", "", "",  # text fields
            None, "",                                            # emotional clip + metrics
            None, "",                                            # emotional‑neutral clip + metrics
            None                                                  # neutral clip (no metrics)
        )

    # ------------------------------------------------------------------
    # Save the uploaded audio so the core can read it
    # ------------------------------------------------------------------
    audio_path = os.path.join(tempfile.gettempdir(), "input.wav")
    try:
        sr, data = audio  # gradio returns (sr, numpy_array)
        sf.write(audio_path, data, sr)
    except Exception as exc:
        logger.error("Error processing audio input: %s", exc)
        return (
            f"Error processing audio: {exc}", "", "", "", "", "", "",  # text fields filled w/ error
            None, "", None, "", None
        )

    # ------------------------------------------------------------------
    # Run the core pipeline
    # ------------------------------------------------------------------
    result = process_audio(audio_path, tts_model_choice, llm_model_choice)

    # We deliberately IGNORE result["neutral_metrics"] (no analysis wanted)

    return (
        result["transcript"],              # 1
        result["asr_metrics"],             # 2
        result["detected_emotion"],        # 3
        result["emotional_response"],      # 4
        result["neutral_response"],        # 5
        result["target_emotion"],          # 6
        result["tts_model"],               # 7
        result["emotional_audio_path"],    # 8  clip 1
        result["emotional_metrics"],       # 9  metrics 1
        result["emotional_neutral_audio_path"],  # 10 clip 2
        result["emotional_neutral_metrics"],     # 11 metrics 2
        result["neutral_audio_path"],      # 12 clip 3 (no metrics)
        result["neutral_metrics"], 
    )


def gradio_text_wrapper(text, llm_model_choice):
    """Handle pure text input."""
    if not text:
        return "", "", ""
    result = process_text(text, llm_model_choice)
    return (
        result["detected_emotion"],
        result["emotional_response"],
        result["neutral_response"],
    )


# ---------------------------------------------------------------------------
# Build the interface
# ---------------------------------------------------------------------------

def run_gradio():
    """Launch the Gradio demo."""

    available_models = ["llama_3_70b_q4", "llama_3_3b_q4", "mistral_7b_q4"]

    with gr.Blocks(title="Emotion‑Aware Speech & Text Demo") as demo:
        gr.Markdown("# Emotion‑Aware Speech & Text Processing Demo")
        gr.Markdown(
            "This demo detects emotion, generates responses, and speaks them back with the desired delivery."
        )

        with gr.Tabs():
            # ----------------------------------------------------------
            # Speech tab
            # ----------------------------------------------------------
            with gr.TabItem("Speech Processing"):
                gr.Markdown("## Speech Processing")

                with gr.Row():
                    # -----------------------------
                    # Input column
                    # -----------------------------
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        audio_input = gr.Audio(label="Speak here", type="numpy")

                        with gr.Row():
                            tts_model_choice = gr.Radio(
                                ["EmotiVoice", "ChatGPT"],
                                label="Select TTS Model",
                                value="EmotiVoice",
                            )
                            llm_model_choice_audio = gr.Dropdown(
                                available_models,
                                label="Select LLM Model",
                                value=DEFAULT_MODEL,
                            )
                        submit_audio_btn = gr.Button("Process Speech", variant="primary")

                    # -----------------------------
                    # Results column
                    # -----------------------------
                    with gr.Column(scale=1):
                        gr.Markdown("### Results")

                        # ASR + detection
                        with gr.Group():
                            gr.Markdown("#### ASR Results")
                            transcript_output = gr.Textbox(label="Transcript")
                            asr_metrics_output = gr.Textbox(label="ASR Metrics")
                            detected_emotion_output = gr.Textbox(label="Detected Emotion")

                        # Responses
                        with gr.Group():
                            gr.Markdown("#### Generated Responses")
                            with gr.Group():
                                gr.Markdown("##### Emotion‑Steered Response")
                                emotional_response_output = gr.Textbox(label="Response Text")
                                target_emotion_output = gr.Textbox(label="Target Emotion")
                                tts_model_output = gr.Textbox(label="TTS Model Used")
                            with gr.Group():
                                gr.Markdown("##### Neutral Response")
                                neutral_response_output = gr.Textbox(label="Response Text")

                        # Audio clips
                        with gr.Group():
                            gr.Markdown("#### Audio Responses")

                            # 1. emotion response + target emotion
                            with gr.Group():
                                gr.Markdown("##### Emotion‑Steered Audio")
                                emotional_audio_output = gr.Audio(label="Audio", type="filepath")
                                emotional_metrics_output = gr.Textbox(label="Emotion2vec Analysis", lines=4, max_lines=10)

                            # 2. emotion response + neutral emotion
                            with gr.Group():
                                gr.Markdown("##### Emotion‑Steered with Neutral Delivery")
                                emotional_neutral_audio_output = gr.Audio(label="Audio", type="filepath")
                                emotional_neutral_metrics_output = gr.Textbox(label="Emotion2vec Analysis", lines=4, max_lines=10)

                            # 3. neutral response + neutral emotion (no metrics)
                            with gr.Group():
                                gr.Markdown("##### Neutral Audio")
                                neutral_audio_output = gr.Audio(label="Audio", type="filepath")
                                neutral_metrics_output = gr.Textbox(label="Emotion2vec Analysis", lines=4, max_lines=10)

            # ----------------------------------------------------------
            # Text tab
            # ----------------------------------------------------------
            with gr.TabItem("Text Processing"):
                gr.Markdown("## Text Processing")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input")
                        text_input = gr.Textbox(label="Enter text here", placeholder="Type your message…", lines=5)
                        llm_model_choice_text = gr.Dropdown(available_models, label="Select LLM Model", value=DEFAULT_MODEL)
                        submit_text_btn = gr.Button("Process Text", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Results")
                        text_emotion_output = gr.Textbox(label="Detected Emotion")
                        with gr.Group():
                            gr.Markdown("#### Generated Responses")
                            with gr.Group():
                                gr.Markdown("##### Emotion‑Steered Response")
                                text_emotional_response = gr.Textbox(label="Response", lines=5)
                            with gr.Group():
                                gr.Markdown("##### Neutral Response")
                                text_neutral_response = gr.Textbox(label="Response", lines=5)

        # --------------------------------------------------------------
        # Wire buttons → callbacks
        # --------------------------------------------------------------
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
                emotional_neutral_audio_output,
                emotional_neutral_metrics_output,
                neutral_audio_output,
                neutral_metrics_output, 
            ],
        )

        submit_text_btn.click(
            fn=gradio_text_wrapper,
            inputs=[text_input, llm_model_choice_text],
            outputs=[text_emotion_output, text_emotional_response, text_neutral_response],
        )

    # ------------------------------------------------------------------
    logger.info("Launching Gradio interface…")
    demo.launch(share=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting application with Gradio interface…")
    print("Launching Gradio interface…")
    run_gradio()

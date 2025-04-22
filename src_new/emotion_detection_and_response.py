#!/usr/bin/env python3
"""
Emotion Detection and Response Core Module
=========================================
This module contains the core functionality for emotion detection and response generation.

Key Functions:
1. detect_emotion - Predicts the most natural emotional response
2. generate_emotional_response - Creates responses with specific emotional tones
3. generate_neutral_response - Creates neutral responses for comparison

The module depends on emotion_utils.py for utilities, constants, and logging.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union, Any

# Import utilities and constants from the utils module
from src_new.emotion_utils import (
    setup_logging,
    map_emotion,
    standardize_reference_emotion,
    LoggedVLLMChatCompletion,
    create_logged_inference_client,
    create_inference_client,
    build_classification_prompt,
    build_generation_prompt,
    build_baseline_prompt,
    ALLOWED_EMOTIONS,
    EMOTION_MAPPING,
    MAX_TOKENS_CLASS,
    MAX_TOKENS_GEN,
    TEMPERATURE_CLASS,
    TEMPERATURE_GEN_DEFAULT,
    LENGTH_RATIO_DEFAULT
)

# Get logger from utils
logger = setup_logging()

# ------------------- Core functionality --------------------

def detect_emotion(text: str, model: str = "llama_3_70b_q4") -> str:
    """
    Detect the most natural emotional response to the given text.
    
    Args:
        text: The text to analyze
        model: The model to use for classification (default: "llama_3_70b_q4")
        
    Returns:
        The predicted emotion as a string
    """
    client = create_logged_inference_client()
    if not client:
        logger.warning("Inference client not available, returning default emotion 'Neutral'")
        return "Neutral"
    
    # Build the classification prompt
    prompt = build_classification_prompt(text)
    
    # Get the predicted emotion
    try:
        predicted_emotion = client.inference_call(
            model, 
            prompt, 
            max_tokens=MAX_TOKENS_CLASS, 
            temperature=TEMPERATURE_CLASS
        ).strip()
        
        # Map the predicted emotion
        mapped_emotion = map_emotion(predicted_emotion)        
        # Validate the emotion
        if mapped_emotion not in ALLOWED_EMOTIONS:
            logger.warning(f"Predicted emotion '{mapped_emotion}' not in allowed emotions, using 'Neutral'")
            mapped_emotion = "Neutral"
        
        return mapped_emotion
    except Exception as e:
        logger.error(f"Error predicting emotion: {e}")
        return "Neutral"

def generate_emotional_response(
    text: str, 
    target_emotion: str, 
    length_ratio: float = LENGTH_RATIO_DEFAULT,
    model: str = "llama_3_70b_q4",
    temperature: float = TEMPERATURE_GEN_DEFAULT,
    example_emotion: str = None
) -> str:
    """
    Generate an emotional response to the given text.
    
    Args:
        text: The text to respond to
        target_emotion: The target emotion for the response
        length_ratio: The ratio of response length to input length (default: 0.9)
        model: The model to use for generation (default: "llama_3_70b_q4")
        temperature: The temperature for generation (default: 0.7)
        example_emotion: Override emotion for selecting few-shot example (default: None = use target_emotion)
        
    Returns:
        The generated response as a string
    """
    client = create_logged_inference_client()
    if not client:
        logger.warning("Inference client not available, returning default response")
        return f"I would respond to '{text}' with {target_emotion} emotion, but the inference client is not available."
    
    # Map the target emotion
    mapped_emotion = map_emotion(target_emotion)
    
    # Calculate target word count
    words_in_text = len(text.split())
    target_words = max(60, int(words_in_text * length_ratio))
    
    # Build the generation prompt with potentially different example emotion
    prompt = build_generation_prompt(text, mapped_emotion, target_words, example_emotion)
    
    # Generate the response
    try:
        response = client.inference_call(
            model, 
            prompt, 
            max_tokens=MAX_TOKENS_GEN, 
            temperature=temperature
        ).strip()

        if "INPUT:" in response:
            response = response.split("INPUT:")[0].strip()
        elif "EMOTION=" in response:
            response = response.split("EMOTION=")[0].strip()
        
        return response.replace("\n", " ")
    except Exception as e:
        logger.error(f"Error generating emotional response: {e}")
        return f"Error generating {mapped_emotion} response: {e}"

def generate_neutral_response(
    text: str, 
    length_ratio: float = LENGTH_RATIO_DEFAULT,
    model: str = "llama_3_70b_q4",
    temperature: float = TEMPERATURE_GEN_DEFAULT
) -> str:
    """
    Generate a neutral response to the given text.
    
    Args:
        text: The text to respond to
        length_ratio: The ratio of response length to input length (default: 0.9)
        model: The model to use for generation (default: "llama_3_70b_q4")
        temperature: The temperature for generation (default: 0.7)
        
    Returns:
        The generated response as a string
    """
    client = create_logged_inference_client()
    if not client:
        logger.warning("Inference client not available, returning default response")
        return f"I would respond to '{text}' neutrally, but the inference client is not available."
    
    # Calculate target word count
    words_in_text = len(text.split())
    target_words = max(60, int(words_in_text * length_ratio))
    
    # Build the baseline prompt with just the 2 examples we've kept
    prompt = build_baseline_prompt(text, target_words)
    
    # Generate the response
    try:
        response = client.inference_call(
            model, 
            prompt, 
            max_tokens=MAX_TOKENS_GEN, 
            temperature=temperature
        ).strip()

        if "INPUT:" in response:
            response = response.split("INPUT:")[0].strip()
        
        return response.replace("\n", " ")
    except Exception as e:
        logger.error(f"Error generating neutral response: {e}")
        return f"Error generating neutral response: {e}"

# ------------------- Simplified functions for Gradio --------------------

def process_for_gradio(
    text: str,
    speaker_emotion: str = "Neutral",
    model: str = "llama_3_70b_q4",
    temperature: float = TEMPERATURE_GEN_DEFAULT,
    length_ratio: float = LENGTH_RATIO_DEFAULT,
    example_emotion: str = None
) -> Dict[str, str]:
    """
    Process text for Gradio interface - detect emotion and generate responses.
    
    Args:
        text: The input text
        speaker_emotion: The emotion of the speaker (default: "Neutral")
        model: The model to use (default: "llama_3_70b_q4")
        temperature: The temperature for generation (default: 0.7)
        length_ratio: The ratio of response length to input length (default: 0.9)
        example_emotion: Optional override for the few-shot example emotion (default: None = use detected emotion)
        
    Returns:
        A dictionary with the detected emotion and responses
    """
    # Detect emotion
    detected_emotion = detect_emotion(text, model)
    
    # Generate emotional response with custom example emotion if provided
    emotional_response = generate_emotional_response(
        text, 
        detected_emotion, 
        length_ratio, 
        model, 
        temperature,
        example_emotion
    )
    
    # Generate neutral response
    neutral_response = generate_neutral_response(
        text, 
        length_ratio, 
        model, 
        temperature
    )
    
    return {
        "detected_emotion": detected_emotion,
        "emotional_response": emotional_response,
        "neutral_response": neutral_response,
    }

# ------------------- Main --------------------

if __name__ == "__main__":
    # Example usage
    print("Emotion Detection and Response Module")
    print("=====================================")
    
    example_text = "I just got a promotion at work today!"
    
    print(f"Input: {example_text}")
    
    # Test emotion detection
    emotion = detect_emotion(example_text)
    print(f"Detected emotion: {emotion}")
    
    # Test response generation
    response = generate_emotional_response(example_text, emotion)
    print(f"\nEmotional response ({emotion}):")
    print(response)
    
    # Test neutral response
    neutral = generate_neutral_response(example_text)
    print(f"\nNeutral response:")
    print(neutral)
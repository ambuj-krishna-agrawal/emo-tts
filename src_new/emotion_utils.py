#!/usr/bin/env python3
"""
Emotion Detection and Response Utilities Module
===============================================
This module provides utilities, constants, and helpers for emotion detection 
and response generation functionality.

Key Components:
1. Logging setup and utilities
2. Emotion constants and mapping functions
3. Prompt building functions
4. Model router integration

This module supports emotion_core.py which contains the main functionality.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from textwrap import dedent

# Try importing VLLMChatCompletion
try:
    from src_new.vllm_router import RawTemplateBuilder, VLLMChatCompletion
except ImportError:
    logging.warning("Failed to import vllm_router - inference client not available")
    RawTemplateBuilder = None
    VLLMChatCompletion = None

# ------------------- Logging --------------------

def setup_logging(log_file: str | Path = "emotion_detection.log") -> logging.Logger:
    """
    Set up logging for the emotion detection module.
    
    Args:
        log_file: The path to the log file (default: "emotion_detection.log")
        
    Returns:
        A configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("emotion_detection")

logger = setup_logging()

# ------------------- LLM Request/Response Logging --------------------

class LLMLogger:
    """Logger for LLM requests and responses in JSON format"""
    
    def __init__(self, log_file: str | Path = "llm_logs.json"):
        self.log_file = Path(log_file)
        # Create the file if it doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write('[]')
    
    def log_call(self, model: str, prompt: List[Dict[str, str]], response: str, 
                 params: Dict[str, Any], duration_ms: float, 
                 call_type: str = "inference") -> None:
        """
        Log an LLM call to the JSON log file
        
        Args:
            model: The model used
            prompt: The prompt sent to the model
            response: The response from the model
            params: Additional parameters for the call
            duration_ms: The duration of the call in milliseconds
            call_type: Type of the call (default: "inference")
        """
        # Clean the response by removing everything after "INPUT:" if present
        cleaned_response = response.split("INPUT:", 1)[0].strip() if "INPUT:" in response else response
        
        # Create log entry with cleaned response
        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "call_type": call_type,
            "model": model,
            "prompt": prompt,
            "response": cleaned_response,
            "parameters": params,
            "duration_ms": round(duration_ms, 2)
        }
        
        # Read existing logs
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logs = []
        
        # Append new log
        logs.append(log_entry)
        
        # Write back to file with pretty formatting
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Logged {call_type} call to {model}, duration: {duration_ms:.2f}ms")

# Initialize the LLM logger
llm_logger = LLMLogger()

# ------------------- Constants --------------------

MAX_TOKENS_CLASS = 10
MAX_TOKENS_GEN = 400
TEMPERATURE_CLASS = 0.0
TEMPERATURE_GEN_DEFAULT = 0.7
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
    "surprised": "Surprise",
    "disgust": "Disgust",
    "disgusted": "Disgust",
    "fear": "Fear",
    "fearful": "Fear",
    "afraid": "Fear",
    "joy": "Happy",
    "joyful": "Happy",
    "excited": "Excited",
    "exciting": "Excited",
    # --- curiosity family mapped to Excited (positive engagement) ---
    "curious": "Excited",
    "curiosity": "Excited",
    "curious to dive deeper": "Excited",
    "interest": "Excited",
    "interested": "Excited",
}

# Adding REFERENCE_EMOTION_MAPPING for compatibility
REFERENCE_EMOTION_MAPPING = EMOTION_MAPPING.copy()

# ------------------- Utility functions --------------------

def map_emotion(raw_input: str) -> str:
    """
    Intelligently map raw emotion text to standardized emotion labels.
    
    This function handles various text formats including sentences, punctuation,
    and extracts the core emotion from complex responses.
    
    Args:
        raw_input: Raw emotion text that might include punctuation or be part of a sentence
        
    Returns:
        A standardized emotion label from the ALLOWED_EMOTIONS set
    """
    if not raw_input:
        return "Neutral"
    
    # Clean the input - remove punctuation and convert to lowercase
    # First check if any of our emotion keywords are in the input
    clean_text = raw_input.lower().strip()
    
    # Remove punctuation and extra whitespace
    clean_text = re.sub(r'[^\w\s]', '', clean_text).strip()
    
    # Try direct mapping first
    if clean_text in EMOTION_MAPPING:
        return EMOTION_MAPPING[clean_text]
    
    # Look for emotion words in the text
    for emotion_key, emotion_value in EMOTION_MAPPING.items():
        # Check if the emotion key is a whole word in the text
        if re.search(r'\b' + re.escape(emotion_key) + r'\b', clean_text):
            return emotion_value
    
    # If we have a single word, try to title-case it and check if it's in ALLOWED_EMOTIONS
    if len(clean_text.split()) == 1:
        titled = clean_text.title()
        if titled in ALLOWED_EMOTIONS:
            return titled
    
    # Extract the most likely emotion word by checking each word
    words = clean_text.split()
    for word in words:
        # Check if the word is similar to any emotion in the mapping
        for emotion_key, emotion_value in EMOTION_MAPPING.items():
            if word == emotion_key or word.startswith(emotion_key):
                return emotion_value
    
    # If all else fails, check if the first word is a valid emotion when title-cased
    if words and words[0].title() in ALLOWED_EMOTIONS:
        return words[0].title()
    
    # Default to Neutral if we couldn't find a match
    logger.warning(f"Could not map emotion from text: '{raw_input}', defaulting to Neutral")
    return "Neutral"

# Adding standardize_reference_emotion for compatibility
def standardize_reference_emotion(raw_input: str) -> str:
    """Alias for map_emotion to maintain compatibility with existing code"""
    return map_emotion(raw_input)

# ------------------- Prompt builders --------------------

def build_classification_prompt(text: str) -> List[Dict[str, str]]:
    """
    Build a prompt to classify the emotional response to a given text.
    
    Args:
        text: The input text to classify
        
    Returns:
        A list of dictionaries representing the prompt for classification
    """
    prompt = [
        {"role": "system", "content": "You are an emotion classification system. Based on Speaker A's text, predict the most natural emotional response for Speaker B. Choose from: Happy, Sad, Angry, Neutral, Surprise, Disgust, Fear, Excited."},
        {"role": "user", "content": f"Speaker A: {text}\n\nWhat emotion would be the most natural for Speaker B to express in response? Answer with a single word."}
    ]
    return prompt

# Simplified style guide template without special characters
LONG_STYLE_GUIDE_TEMPLATE = (
    "You are Speaker B. Respond with the target EMOTION in 100 words or less. "
    "Make the emotion obvious. Force yourself to reply in specified emotion ONLY. "
    "Refer to something specific the speaker just said. "
    "End with a gentle follow-up question."
)

# Few-shot examples for emotional responses - cleaned up with no special characters or linebreaks
FEWSHOT_LONG = [
    # SURPRISE
    (
        "I didn't expect you to come!\nEMOTION=Surprise",
        "Whoa, I didn't see that coming! It's such a treat to bump into you. Days like this remind me how fun surprises are. Care to grab a coffee and tell me what sparked this visit?"
    ),
    # FEAR
    (
        "The lights just went out…\nEMOTION=Fear",
        "Hey, it's okay I'm right here. Take a breath; the power will likely be back soon. My brother and I used to tell silly stories in the dark to stay calm. Want me to keep chatting until the lights return?"
    ),
    # HAPPY
    (
        "I got the job offer today!\nEMOTION=Happy",
        "That's phenomenal news congratulations! You worked so hard, and I remember you saying the team felt like a perfect fit. How will you celebrate this big win tonight?"
    ),
    # SAD
    (
        "I failed the exam again.\nEMOTION=Sad",
        "I'm really sorry that must feel discouraging. Your effort hasn't gone unnoticed. When I hit a wall last term, talking it through helped. Would it help to unpack what tripped you up?"
    ),
    # ANGRY
    (
        "You broke my phone.\nEMOTION=Angry",
        "Seriously? My phone was brand new. Now I'm stuck without my main lifeline, and it's a real cost. What do you suggest we do to make this right?"
    ),
    # DISGUST
    (
        "This smells awful.\nEMOTION=Disgust",
        "Ugh, that stench is turning my stomach. Let's seal it up and get fresh air in before it spreads. Can you grab a bag so we can toss it out?"
    ),
    # EXCITED
    (
        "We can finally travel next month!\nEMOTION=Excited",
        "Yes! I've been day dreaming about this trip for ages. Imagining us strolling new streets gives me butterflies. Which city are you most pumped about?"
    ),
    # NEUTRAL
    (
        "I'll call you later about the plan.\nEMOTION=Neutral",
        "Sounds good I'll keep my phone handy. Once we connect we can walk through the timeline step by step. Talk soon; let me know if anything changes?"
    ),
]

def build_generation_prompt(history: str, emotion: str, target_words: int = 100, top_n_fewshot=None) -> List[Dict[str, str]]:
    """
    Build a prompt for generating an emotion-specific response.
    
    Args:
        history: The conversation history or utterance to respond to
        emotion: The target emotion for the response
        target_words: The target word count for the response
        top_n_fewshot: If provided, use this as the target emotion for selecting the example
        
    Returns:
        A list of dictionaries representing the prompt for generation
    """
    # Add explicit instruction to only generate one response
    guide = LONG_STYLE_GUIDE_TEMPLATE + " Generate ONLY ONE response to this input."
    
    shots: list[dict[str, str]] = [
        {"role": "system", "content": f"You are Speaker B. Target emotion = **{emotion}**. {guide}"},
    ]

    # Find the appropriate example for the target emotion (or override)
    target = emotion if top_n_fewshot is None else top_n_fewshot
    selected_example = None
    
    # Find the example that matches the target emotion
    for example in FEWSHOT_LONG:
        history_part = example[0]
        if f"EMOTION={target}" in history_part:
            selected_example = example
            break
    
    # If we found a matching example, use it
    if selected_example:
        hist, reply = selected_example
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})
    
    # Add a clear marker to indicate this is the input to respond to
    shots.append({"role": "user", "content": f"INPUT: {history}\nEMOTION={emotion}"})
    
    return shots

# Simplified baseline guide template without special characters
BASELINE_GUIDE_TEMPLATE = (
    "Reply in under 100 words in a Neutral tone regardless of the emotional content of the input. "
    "Be polite, factual, and keep the conversation moving with a simple follow-up question at the end. "
    "Maintain a neutral tone even if the input suggests sadness, excitement, anger, or other emotions. "
    "Generate ONLY ONE response to the input marked as 'INPUT:' below."
)

# Simplified neutral examples with no special characters
NEUTRAL_FEWSHOT = [
    ("INPUT: The meeting starts at nine.", "Got it, thanks for the heads up. I'll log in a few minutes early to ensure the slides load properly. Sound good?"),
    ("INPUT: I'm so devastated. My dog just died after 12 years together.", "I understand this is a significant loss. Many people find comfort in creating a small memorial or sharing favorite memories during this time. Would it help to talk about some special moments you shared together?")
]

def build_baseline_prompt(history: str, target_words: int = 100) -> List[Dict[str, str]]:
    """
    Build a prompt for generating a neutral response regardless of input emotion.
    
    Args:
        history: The conversation history or utterance to respond to
        target_words: The target word count for the response
        
    Returns:
        A list of dictionaries representing the prompt for generation
    """
    guide = BASELINE_GUIDE_TEMPLATE
    shots = [{"role": "system", "content": guide}]

    # Always use our carefully selected examples
    for hist, reply in NEUTRAL_FEWSHOT:
        shots.append({"role": "user", "content": hist})
        shots.append({"role": "assistant", "content": reply})
    
    # Use a clear marker to indicate this is the input to respond to
    shots.append({"role": "user", "content": f"INPUT: {history}\nEMOTION=Neutral"})
    
    return shots

# ------------------- Model router integration --------------------

def get_model_configs():
    """Get model configurations"""
    return {
        "llama_3_70b_q4": {"model_id": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", "base_url": "http://babel-7-25:8081/v1", "is_chat": False, "template_builder": None},
        "llama_3_3b_q4": {"model_id": "AMead10/Llama-3.2-3B-Instruct-AWQ", "base_url": "http://babel-0-35:8083/v1", "is_chat": False, "template_builder": None},
        "mistral_7b_q4": {"model_id": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ", "base_url": "http://babel-7-17:8082/v1", "is_chat": False, "template_builder": None},
    }

def create_inference_client():
    """Create and return the VLLMChatCompletion client"""
    try:
        if VLLMChatCompletion is None:
            logger.error("VLLMChatCompletion is not available")
            return None
            
        cfgs = get_model_configs()
        for cfg in cfgs.values():
            cfg["template_builder"] = RawTemplateBuilder()
        return VLLMChatCompletion(cfgs)
    except ImportError:
        logger.error("Failed to import vllm_router - inference client not available")
        return None

# ------------------- Enhanced VLLMChatCompletion client --------------------

class LoggedVLLMChatCompletion:
    """A wrapper around VLLMChatCompletion that logs all inference calls"""
    
    def __init__(self, client):
        self.client = client
        
    def inference_call(self, model, prompt, max_tokens=None, temperature=None):
        """
        Make an inference call and log it
        
        Args:
            model: The model to use
            prompt: The prompt to send
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            
        Returns:
            The response from the model
        """
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Record start time
        start_time = time.time()
        
        # Make the call
        response = self.client.inference_call(
            model, 
            prompt, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        # Clean the response - remove everything after "INPUT:" if present
        cleaned_response = response.split("INPUT:", 1)[0].strip() if "INPUT:" in response else response
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log the call
        llm_logger.log_call(
            model=model,
            prompt=prompt,
            response=cleaned_response,
            params=params,
            duration_ms=duration_ms
        )
        
        return cleaned_response

def create_logged_inference_client():
    """Create and return a logged VLLMChatCompletion client"""
    client = create_inference_client()
    if client:
        return LoggedVLLMChatCompletion(client)
    return None
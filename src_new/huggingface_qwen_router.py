#!/usr/bin/env python3
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor
)
import logging
from typing import Dict, Any, List, Optional
import traceback
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set HuggingFace cache directories
os.environ['HF_HOME'] = '/data/hf_cache/ambuja'
os.environ['TRANSFORMERS_CACHE'] = '/data/hf_cache/ambuja'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/hf_cache/ambuja'
os.environ["HF_HUB_DOWNLOAD_PARAMS"] = '{"cache_dir": "/data/hf_cache/ambuja"}'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class RawTemplateBuilder:
    """Passes through messages without modification"""
    def build(self, prompt_messages, tokenizer=None):
        return prompt_messages

class HuggingFaceLLMChatCompletion:
    """Handles inference with HuggingFace Qwen models locally"""
    
    def __init__(self, model_configs: Dict):
        self.model_configs = model_configs
        self.models = {}
        self.tokenizers = {}
        self.processors = {}  # Added processors dictionary
        self._initialize_models()

    def _initialize_models(self):
        """Initialize models with automatic multi-GPU distribution"""
        for model_alias, cfg in self.model_configs.items():
            logger.info(f"Initializing model {model_alias} from {cfg['name']}")
            
            try:
                # Configure quantization for efficient loading
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                
                # Load model and tokenizer
                if "qwen2-vl" in cfg["name"].lower():  # More specific check for VL models
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        cfg["name"],
                        quantization_config=quant_config,
                        device_map="cuda:0",  # Or specify exact GPU IDs if multiple
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        cache_dir="/data/hf_cache/ambuja"
                    )
                    
                    processor = Qwen2VLProcessor.from_pretrained(
                        cfg["name"],
                        trust_remote_code=True,
                        cache_dir="/data/hf_cache/ambuja"
                    )
                    
                    # Store processor instead of tokenizer for VL models
                    self.processors[model_alias] = processor
                    # Use the tokenizer from the processor for text processing
                    self.tokenizers[model_alias] = processor.tokenizer
                    
                elif "qwen2" in cfg["name"].lower():  # Regular Qwen2 text models
                    model = AutoModelForCausalLM.from_pretrained(
                        cfg["name"],
                        quantization_config=quant_config,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        cache_dir="/data/hf_cache/ambuja"
                    )
                    
                    tokenizer = Qwen2Tokenizer.from_pretrained(
                        cfg["name"],
                        trust_remote_code=True,
                        cache_dir="/data/hf_cache/ambuja"
                    )
                    
                    # Ensure tokenizer has necessary tokens
                    if not tokenizer.pad_token:
                        if tokenizer.eos_token:
                            tokenizer.pad_token = tokenizer.eos_token
                        else:
                            tokenizer.pad_token = tokenizer.eos_token = "</s>"
                            
                    self.tokenizers[model_alias] = tokenizer
                else:
                    # Generic loading for other models
                    model = AutoModelForCausalLM.from_pretrained(
                        cfg["name"],
                        quantization_config=quant_config,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        cache_dir="/data/hf_cache/ambuja"
                    )
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        cfg["name"],
                        trust_remote_code=True,
                        cache_dir="/data/hf_cache/ambuja"
                    )
                    
                    # Ensure tokenizer has necessary tokens
                    if not tokenizer.pad_token:
                        if tokenizer.eos_token:
                            tokenizer.pad_token = tokenizer.eos_token
                        else:
                            tokenizer.pad_token = tokenizer.eos_token = "</s>"
                    
                    self.tokenizers[model_alias] = tokenizer
                
                # Store models
                self.models[model_alias] = model
                logger.info(f"Successfully loaded model {model_alias}")
                
            except Exception as e:
                logger.error(f"Error loading model {model_alias}: {e}")
                logger.error(traceback.format_exc())
                raise

    def _get_model_components(self, model_alias: str):
        """Get model and tokenizer for a specific alias"""
        if model_alias not in self.models or model_alias not in self.tokenizers:
            raise ValueError(f"No loaded model for alias: {model_alias}")
        return self.models[model_alias], self.tokenizers[model_alias]

    def _process_audio(self, audio_path):
        """Load and process audio file for model input"""
        try:
            logger.info(f"Loading audio from: {audio_path}")
            # Load audio using librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            # Convert to float32 for model compatibility
            audio = audio.astype(np.float32)
            return audio
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _extract_audio_tags(self, messages):
        """Extract audio paths from <audio> tags in messages and replace with placeholders"""
        processed_messages = []
        audio_paths = []
        
        for message in messages:
            content = message.get("content", "")
            if "<audio>" in content and "</audio>" in content:
                # Extract audio path
                start_idx = content.find("<audio>") + len("<audio>")
                end_idx = content.find("</audio>")
                audio_path = content[start_idx:end_idx].strip()
                audio_paths.append(audio_path)
                
                # Replace tag with placeholder for text-only processing
                new_content = content.replace(f"<audio>{audio_path}</audio>", "[AUDIO]")
                processed_message = message.copy()
                processed_message["content"] = new_content
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)
                
        return processed_messages, audio_paths

    def inference_call(
        self,
        model_alias: str,
        prompt_messages: List[Dict[str, Any]],
        max_tokens: int = 100,
        temperature: float = 0.0
    ) -> str:
        """Generate a response for the provided messages with audio processing capability"""
        try:
            # Get model and tokenizer
            model, tokenizer = self._get_model_components(model_alias)
            
            # Get template builder from config
            cfg = self.model_configs.get(model_alias)
            if not cfg:
                raise ValueError(f"Model configuration not found for alias: {model_alias}")
            
            # Extract audio paths from messages
            processed_messages, audio_paths = self._extract_audio_tags(prompt_messages)
            
            # Apply template builder if provided
            builder = cfg.get("template_builder")
            messages = builder.build(processed_messages, tokenizer) if builder else processed_messages
            
            logger.info(f"Processing messages for model {model_alias}")
            
            # Handle different input formats based on model type
            if "qwen2-vl" in cfg["name"].lower() and audio_paths:
                # Use processor for multimodal inputs
                processor = self.processors.get(model_alias)
                if not processor:
                    raise ValueError(f"No processor found for multimodal model {model_alias}")
                
                # Load and process audio
                audio_data = [self._process_audio(path) for path in audio_paths]
                
                # Process text and audio together
                text_input = "\n".join(m.get("content", "") for m in messages if m.get("content"))
                inputs = processor(text=text_input, audio=audio_data, return_tensors="pt")
                
            elif cfg.get("is_chat", True):  # Default to chat format for text models
                # Format as chat for chat models
                input_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize input
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_attention_mask=True
                )
            else:
                # Format as text completion for non-chat models
                input_text = "\n".join(m.get("content", "") for m in messages)
                
                # Tokenize input
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_attention_mask=True
                )
            
            # Move to GPU
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate output
            with torch.no_grad(), torch.amp.autocast("cuda"):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,  # Use sampling only if temperature > 0
                    temperature=max(temperature, 1e-5),  # Prevent division by zero
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Trim input tokens from output and decode
            input_length = inputs["input_ids"].shape[1]
            trimmed_ids = generated_ids[0, input_length:]
            output_text = tokenizer.decode(
                trimmed_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            logger.info(f"Successfully generated response for {model_alias}")
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Error during inference with model {model_alias}: {e}")
            logger.error(traceback.format_exc())
            raise
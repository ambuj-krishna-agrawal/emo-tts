#!/usr/bin/env python3
import torch
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")

def format_message(role: str, content: str) -> Dict[str, str]:
    """Format a message for chat models"""
    return {"role": role, "content": content}

def build_chat_messages(system_prompt: str, user_content: str) -> List[Dict[str, str]]:
    """Build a simple chat message sequence with system and user messages"""
    messages = []
    if system_prompt:
        messages.append(format_message("system", system_prompt))
    messages.append(format_message("user", user_content))
    return messages

def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """Get memory information for all available GPUs"""
    if not torch.cuda.is_available():
        return {}
    
    info = {}
    for i in range(torch.cuda.device_count()):
        free_mem = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
        
        info[i] = {
            "device_name": torch.cuda.get_device_name(i),
            "free_memory_gb": free_mem,
            "total_memory_gb": total_mem,
            "used_percentage": (free_mem / total_mem) * 100
        }
    
    return info

def log_gpu_status():
    """Log current GPU memory status"""
    if torch.cuda.is_available():
        info = get_gpu_memory_info()
        for gpu_id, stats in info.items():
            logger.info(f"GPU {gpu_id} ({stats['device_name']}): "
                        f"{stats['free_memory_gb']:.2f} GB used / {stats['total_memory_gb']:.2f} GB "
                        f"({stats['used_percentage']:.2f}%)")
    else:
        logger.info("No GPU available")
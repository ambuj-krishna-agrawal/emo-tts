#!/usr/bin/env python3
"""
bpe_model_setup.py
Download and set up the BPE model files needed for ESPnet ASR.

This script downloads the necessary BPE model files manually and places them
in the expected directory structure.
"""

import os
import requests
import logging
from pathlib import Path
import tarfile
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bpe_model_setup")

# Constants
BPE_MODEL_URL = "https://zenodo.org/record/3969130/files/data.tar.gz"
DATA_DIR = Path("data")
TOKEN_LIST_DIR = DATA_DIR / "token_list" / "bpe_unigram5000"

def download_file(url, target_path):
    """Download a file from URL to target path."""
    logger.info(f"Downloading {url} to {target_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return target_path

def extract_tar(tar_path, extract_dir):
    """Extract a tar file to the specified directory."""
    logger.info(f"Extracting {tar_path} to {extract_dir}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)

def main():
    """Main function to set up BPE model files."""
    logger.info("Starting BPE model setup")
    
    # Create directories
    TOKEN_LIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download the data archive
    temp_dir = Path("temp_download")
    temp_dir.mkdir(exist_ok=True)
    tar_path = temp_dir / "data.tar.gz"
    
    try:
        # Download the model file archive
        download_file(BPE_MODEL_URL, tar_path)
        
        # Extract the archive
        extract_tar(tar_path, temp_dir)
        
        # Check for BPE model files
        bpe_files = list(temp_dir.glob("**/model.model"))
        if not bpe_files:
            logger.error("No BPE model files found in the downloaded archive")
            return
        
        # Copy the BPE model file to the expected location
        bpe_source = bpe_files[0]
        bpe_target = TOKEN_LIST_DIR / "model.model"
        shutil.copy(bpe_source, bpe_target)
        logger.info(f"Copied BPE model from {bpe_source} to {bpe_target}")
        
        # Copy vocab file if it exists
        vocab_files = list(temp_dir.glob("**/vocab.txt"))
        if vocab_files:
            vocab_source = vocab_files[0]
            vocab_target = TOKEN_LIST_DIR / "vocab.txt"
            shutil.copy(vocab_source, vocab_target)
            logger.info(f"Copied vocab from {vocab_source} to {vocab_target}")
        
        logger.info("BPE model setup complete")
        
    except Exception as e:
        logger.error(f"Error setting up BPE model: {e}")
    
    finally:
        # Clean up temporary files
        if tar_path.exists():
            tar_path.unlink()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
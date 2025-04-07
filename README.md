# ESPnet TTS Inference on IEMOCAP

This repository contains a script for synthesizing speech using a pre‑trained [ESPnet](https://github.com/espnet/espnet) text-to-speech (TTS) model with the [IEMOCAP dataset](https://huggingface.co/datasets/AbstractTTS/IEMOCAP) from Hugging Face.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
   1. [Clone the Repository](#clone-the-repository)
   2. [Create the Conda Environment](#create-the-conda-environment)
   3. [Activate the Environment](#activate-the-environment)
   4. [Install Additional Dependencies](#install-additional-dependencies)
3. [Usage](#usage)
4. [How It Works](#how-it-works)
5. [Troubleshooting](#troubleshooting)
6. [License](#license)

## Overview

- **Purpose:**  
  The script loads the IEMOCAP dataset and synthesizes speech for each entry using a pre‑trained ESPnet TTS model. It outputs WAV files and generates a metadata JSON file for later evaluation.

- **Features:**  
  - Downloads the dataset from Hugging Face using a provided authentication token.
  - Uses the `Text2Speech` class from ESPnet to generate speech.
  - Adjusts the sample rate if necessary.
  - Saves generated audio files in an organized directory structure.
  - Creates a `metadata.json` file containing details (utterance IDs, transcripts, and file paths) for each sample.

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Create the Conda Environment

Use the provided environment file (environment_detailed.json) to create the environment with the exact state:

```bash
conda env create --file environment_detailed.json
```

**Note:**
Exporting the environment with `conda env export --json > environment_detailed.json` captures the complete state (packages, versions, channels, and build details) of your current environment. This file should allow you to recreate the same environment on another system—as long as the packages remain available. Some users prefer exporting to YAML for broader compatibility, but the JSON export is suitable for exact replication if used promptly.

### 3. Activate the Environment

```bash
conda activate espnet-tts
```

### 4. Install Additional Dependencies (if needed)

If you experience issues related to compilers or libraries, you may need to install specific versions of GCC and libstdcxx-ng:

```bash
conda install -c conda-forge gcc=12
conda install -c conda-forge libstdcxx-ng
```

## Usage

Run the inference script with:

```bash
python inference_hf.py --max_samples 100
```

### Command-Line Arguments

- **--model_tag**: (Default: espnet/kan-bayashi_ljspeech_vits)  
  Specify the ESPnet model tag or a local checkpoint.

- **--split**: (Default: train)  
  Dataset split to use (currently, only "train" is supported).

- **--out_dir**: (Default: generated_wavs)  
  Directory where the output WAV files and metadata will be saved.

- **--hf_token**:  
  Your Hugging Face authentication token. If not provided via the command line, the script will try to use the HF_TOKEN environment variable.

- **--sample_rate**: (Default: 16000)  
  Output sample rate for the synthesized audio.

- **--device**: (Choices: cpu, cuda, mps)  
  Device to run the inference on. Defaults to "cuda" if available.

- **--max_samples**: (Default: 100)  
  Maximum number of samples to process (useful for testing).

## How It Works

1. **Model Initialization**:  
   The script loads a pre‑trained ESPnet TTS model and retrieves its native sample rate.

2. **Dataset Loading**:  
   The IEMOCAP dataset is loaded from Hugging Face using the provided token. The script inspects the dataset to determine the correct field names for the text input.

3. **Speech Synthesis**:  
   For each entry in the dataset, the script synthesizes speech from the text. If necessary, the synthesized audio is resampled to match the specified output sample rate.

4. **Output**:  
   The generated WAV files are saved in the `out_dir/<split>/` folder. A `metadata.json` file is also generated, containing details about each processed sample.

## Troubleshooting

- **Environment Issues**:  
  If you encounter issues related to compilers or libraries, ensure that you have installed the correct versions of gcc and libstdcxx-ng as shown above.

- **Dataset or Token Issues**:  
  Make sure your Hugging Face token is set via the `HF_TOKEN` environment variable or provided via the command line.

## License

[Add your license information here]
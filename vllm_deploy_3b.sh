#!/bin/sh
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --job-name=hermes3_3b_awq
#SBATCH --error=/home/ambuja/logs/error/hermes3_3b_awq.err
#SBATCH --output=/home/ambuja/logs/output/hermes3_3b_awq.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ambuja@andrew.cmu.edu

mkdir -p /scratch/ambuja/model

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_host
export HF_HOME=/home/ambuja/hf_cache
huggingface-cli login --token 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

python -m vllm.entrypoints.openai.api_server \
    --model AMead10/Llama-3.2-3B-Instruct-AWQ \
    --port 8083 \
    --download-dir /scratch/ambuja/model \
    --quantization awq \
    --dtype float16

echo "✅ Hermes‑3‑3B AWQ running on port 8083"

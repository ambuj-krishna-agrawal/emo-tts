#!/bin/sh
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --job-name=llama2_7b_awq
#SBATCH --error=/home/ambuja/logs/error/llama2_7b_awq.err
#SBATCH --output=/home/ambuja/logs/output/llama2_7b_awq.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ambuja@andrew.cmu.edu

mkdir -p /scratch/ambuja/model

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_host
export HF_HOME=/home/ambuja/hf_cache
huggingface-cli login --token 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

python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ\
    --port 8082 \
    --download-dir /scratch/ambuja/model \
    --quantization awq \
    --dtype float16

echo "✅ Llama‑2‑7B‑Chat AWQ running on port 8082"
